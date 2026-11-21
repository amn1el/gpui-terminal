//! Main terminal view component for GPUI.
//!
//! This module provides [`TerminalView`], the primary component for embedding terminals
//! in GPUI applications. It manages:
//!
//! - **I/O Streams**: Accepts arbitrary [`Read`]/[`Write`]
//!   streams, allowing integration with any PTY implementation
//! - **Event Handling**: Keyboard and mouse input, with configurable callbacks
//! - **Rendering**: Efficient canvas-based rendering via [`TerminalRenderer`]
//! - **Configuration**: Font, colors, dimensions, and padding via [`TerminalConfig`]
//!
//! # Architecture
//!
//! The terminal uses a push-based async I/O architecture:
//!
//! 1. A background thread reads bytes from the PTY stdout in 4KB chunks
//! 2. Bytes are sent through a [flume](https://docs.rs/flume) channel to an async task
//! 3. The async task processes bytes through the VTE parser and calls `cx.notify()`
//! 4. GPUI repaints the terminal with the updated grid
//!
//! This approach ensures the terminal only wakes when data arrives, avoiding polling.
//!
//! # Thread Safety
//!
//! - [`TerminalView`] itself is not `Send` (it contains GPUI handles)
//! - The stdin writer is wrapped in `Arc<parking_lot::Mutex<>>` for thread-safe writes
//! - Callbacks ([`ResizeCallback`], [`KeyHandler`]) must be `Send + Sync`
//!
//! # Example
//!
//! ```ignore
//! use gpui::{Context, Edges, px};
//! use gpui_terminal::{ColorPalette, TerminalConfig, TerminalView};
//!
//! // In a GPUI window context:
//! let terminal = cx.new(|cx| {
//!     TerminalView::new(pty_writer, pty_reader, TerminalConfig::default(), cx)
//!         .with_resize_callback(move |cols, rows| {
//!             // Notify PTY of new dimensions
//!         })
//!         .with_exit_callback(|_, cx| {
//!             cx.quit();
//!         })
//! });
//!
//! // Focus the terminal to receive keyboard input
//! terminal.read(cx).focus_handle().focus(window);
//! ```

use alacritty_terminal::Term;
use alacritty_terminal::grid::Dimensions;

use crate::backend::{Edges, prelude::FluentBuilder, *};
use crate::colors::ColorPalette;
use crate::event::{GpuiEventProxy, TerminalEvent};
use crate::input::keystroke_to_bytes;
use crate::render::TerminalRenderer;
use crate::terminal::TerminalState;
use std::io::{Read, Write};
use std::sync::Arc;
use std::sync::mpsc;
use std::thread;

/// Configuration for terminal creation and runtime updates.
///
/// This struct defines the terminal's appearance and behavior, including
/// grid dimensions, font settings, scrollback buffer, and color scheme.
///
/// # Default Values
///
/// | Field | Default |
/// |-------|---------|
/// | `cols` | 80 |
/// | `rows` | 24 |
/// | `font_family` | "monospace" |
/// | `font_size` | 14px |
/// | `scrollback` | 10000 |
/// | `line_height_multiplier` | 1.0 |
/// | `padding` | 0px all sides |
/// | `colors` | Default palette |
/// | `copy_on_selection` | false |
///
/// # Example
///
/// ```ignore
/// use gpui::{Edges, px};
/// use gpui_terminal::{ColorPalette, TerminalConfig};
///
/// let config = TerminalConfig {
///     cols: 120,
///     rows: 40,
///     font_family: "JetBrains Mono".into(),
///     font_size: px(13.0),
///     scrollback: 50000,
///     line_height_multiplier: 1.0,
///     copy_on_selection: true,
///     padding: Edges::all(px(10.0)),
///     colors: ColorPalette::builder()
///         .background(0x1a, 0x1a, 0x1a)
///         .foreground(0xe0, 0xe0, 0xe0)
///         .build(),
/// };
/// ```
///
/// # Runtime Updates
///
/// Configuration can be updated at runtime via [`TerminalView::update_config`].
/// This is useful for implementing features like dynamic font sizing:
///
/// ```ignore
/// terminal.update(cx, |terminal, cx| {
///     let mut config = terminal.config().clone();
///     config.font_size += px(1.0);
///     terminal.update_config(config, cx);
/// });
/// ```
#[derive(Clone, Debug)]
pub struct TerminalConfig {
    /// Number of columns (character width) in the terminal
    pub cols: usize,

    /// Number of rows (lines) in the terminal
    pub rows: usize,

    /// Font family name (e.g., "Fira Code", "JetBrains Mono")
    pub font_family: String,

    /// Font size in pixels
    pub font_size: Pixels,

    /// Maximum number of scrollback lines to keep in history
    pub scrollback: usize,

    /// Multiplier for line height to accommodate tall glyphs (e.g., nerd fonts)
    /// Default is 1.0 (no extra height)
    pub line_height_multiplier: f32,

    /// Padding around the terminal content (top, right, bottom, left)
    /// The padding area renders with the terminal's background color
    pub padding: Edges<Pixels>,

    /// Color palette for terminal colors (16 ANSI colors, 256 extended colors,
    /// foreground, background, and cursor colors)
    pub colors: ColorPalette,

    /// Copy to clipboard on text selection
    pub copy_on_select: bool,
}

impl Default for TerminalConfig {
    fn default() -> Self {
        Self {
            cols: 80,
            rows: 24,
            font_family: "monospace".into(),
            font_size: px(14.0),
            scrollback: 10000,
            line_height_multiplier: 1.0,
            padding: Edges::all(px(0.0)),
            colors: ColorPalette::default(),
            copy_on_select: false,
        }
    }
}

/// Callback type for PTY resize notifications.
///
/// This callback is invoked when the terminal grid dimensions change,
/// typically due to window resizing. The callback receives the new
/// column and row counts.
///
/// # Arguments
///
/// * `cols` - New number of columns (characters wide)
/// * `rows` - New number of rows (lines tall)
///
/// # Thread Safety
///
/// This callback must be `Send + Sync` as it may be called from the render thread.
///
/// # Example
///
/// ```ignore
/// use portable_pty::PtySize;
///
/// let pty = Arc::new(Mutex::new(pty_master));
/// let pty_clone = pty.clone();
///
/// terminal.with_resize_callback(move |cols, rows| {
///     pty_clone.lock().resize(PtySize {
///         cols: cols as u16,
///         rows: rows as u16,
///         pixel_width: 0,
///         pixel_height: 0,
///     }).ok();
/// });
/// ```
pub type ResizeCallback = Box<dyn Fn(usize, usize) + Send + Sync>;

/// Callback type for key event interception.
///
/// This callback is invoked before the terminal processes a key event,
/// allowing you to intercept and handle specific key combinations.
///
/// # Arguments
///
/// * `event` - The key down event from GPUI
///
/// # Returns
///
/// * `true` - Consume the event (terminal will not process it)
/// * `false` - Let the terminal handle the event normally
///
/// # Thread Safety
///
/// This callback must be `Send + Sync`.
///
/// # Example
///
/// ```ignore
/// terminal.with_key_handler(|event| {
///     let keystroke = &event.keystroke;
///
///     // Intercept Ctrl++ for font size increase
///     if keystroke.modifiers.control && (keystroke.key == "+" || keystroke.key == "=") {
///         // Handle font size increase
///         return true; // Consume the event
///     }
///
///     // Intercept Ctrl+- for font size decrease
///     if keystroke.modifiers.control && keystroke.key == "-" {
///         // Handle font size decrease
///         return true;
///     }
///
///     false // Let terminal handle all other keys
/// });
/// ```
pub type KeyHandler = Box<dyn Fn(&KeyDownEvent) -> bool + Send + Sync>;

/// Callback for terminal bell events.
///
/// This callback is invoked when the terminal bell is triggered (BEL character,
/// ASCII 0x07), allowing you to play a sound or show a visual indicator.
///
/// # Arguments
///
/// * `window` - The GPUI window
/// * `cx` - The context for the TerminalView
///
/// # Example
///
/// ```ignore
/// terminal.with_bell_callback(|window, cx| {
///     // Option 1: Visual bell (flash the window or show an indicator)
///     // Option 2: Play a sound
///     // Option 3: Notify the user via system notification
/// });
/// ```
pub type BellCallback = Box<dyn Fn(&mut Window, &mut Context<TerminalView>)>;

/// Callback for terminal title changes.
///
/// This callback is invoked when the terminal title changes via escape sequences
/// (OSC 0, OSC 2), allowing you to update the window or tab title.
///
/// # Arguments
///
/// * `window` - The GPUI window
/// * `cx` - The context for the TerminalView
/// * `title` - The new title string
///
/// # Example
///
/// ```ignore
/// terminal.with_title_callback(|window, cx, title| {
///     // Update the window title
///     // Or update a tab label in a tabbed interface
///     println!("Terminal title changed to: {}", title);
/// });
/// ```
pub type TitleCallback = Box<dyn Fn(&mut Window, &mut Context<TerminalView>, &str)>;

/// Callback for clipboard store requests.
///
/// This callback is invoked when the terminal wants to store data to the clipboard
/// via OSC 52 escape sequence. Applications like tmux and vim can use this to
/// copy text to the system clipboard.
///
/// # Arguments
///
/// * `window` - The GPUI window
/// * `cx` - The context for the TerminalView
/// * `text` - The text to store in the clipboard
///
/// # Example
///
/// ```ignore
/// use gpui_terminal::Clipboard;
///
/// terminal.with_clipboard_store_callback(|window, cx, text| {
///     if let Ok(mut clipboard) = Clipboard::new() {
///         clipboard.copy(text).ok();
///     }
/// });
/// ```
pub type ClipboardStoreCallback = Box<dyn Fn(&mut Window, &mut Context<TerminalView>, &str)>;

/// Callback for clipboard load requests.
///
/// This is invoked when an application inside the terminal wants to paste text
/// via OSC 52. The callback should return the text from the system clipboard,
/// or `None` if the request is denied (e.g., for security reasons).
pub type ClipboardLoadCallback =
    Box<dyn Fn(&mut Window, &mut Context<TerminalView>) -> Option<String>>;

/// Callback for terminal exit events.
///
/// This callback is invoked when the terminal process exits (e.g., shell exits,
/// process terminates). This is detected when the PTY reader reaches EOF.
///
/// # Arguments
///
/// * `window` - The GPUI window
/// * `cx` - The context for the TerminalView
///
/// # Example
///
/// ```ignore
/// terminal.with_exit_callback(|window, cx| {
///     // Option 1: Quit the application
///     cx.quit();
///
///     // Option 2: Close this terminal tab/pane
///     // terminal_manager.close_terminal(terminal_id);
///
///     // Option 3: Show an exit message
///     // show_notification("Terminal exited");
/// });
/// ```
pub type ExitCallback = Box<dyn Fn(&mut Window, &mut Context<TerminalView>)>;

/// The main terminal view component for GPUI applications.
///
/// `TerminalView` is a GPUI entity that implements the [`Render`] trait,
/// providing a complete terminal emulator that can be embedded in any GPUI application.
///
/// # Responsibilities
///
/// - **Terminal State**: Manages the grid, cursor, and colors via [`TerminalState`]
/// - **I/O Streams**: Reads from PTY stdout and writes to PTY stdin
/// - **Event Handling**: Processes keyboard, mouse, and resize events
/// - **Rendering**: Paints text, backgrounds, and cursor via [`TerminalRenderer`]
/// - **Callbacks**: Dispatches events to user-provided callbacks
///
/// # Creating a Terminal
///
/// Use [`TerminalView::new`] within a GPUI entity context:
///
/// ```ignore
/// let terminal = cx.new(|cx| {
///     TerminalView::new(writer, reader, config, cx)
///         .with_resize_callback(resize_callback)
///         .with_exit_callback(|_, cx| cx.quit())
/// });
/// ```
///
/// # Focus
///
/// The terminal must be focused to receive keyboard input:
///
/// ```ignore
/// terminal.read(cx).focus_handle().focus(window);
/// ```
///
/// # Callbacks
///
/// Configure behavior through builder methods:
///
/// - [`with_resize_callback`](Self::with_resize_callback) - PTY size changes
/// - [`with_exit_callback`](Self::with_exit_callback) - Process exit
/// - [`with_key_handler`](Self::with_key_handler) - Key event interception
/// - [`with_bell_callback`](Self::with_bell_callback) - Terminal bell
/// - [`with_title_callback`](Self::with_title_callback) - Title changes
/// - [`with_clipboard_store_callback`](Self::with_clipboard_store_callback) - Clipboard writes
///
/// # Thread Safety
///
/// `TerminalView` is not `Send` as it contains GPUI handles. The stdin writer
/// is internally wrapped in `Arc<parking_lot::Mutex<>>` for safe concurrent access.
pub struct TerminalView {
    /// The terminal state managing the grid and VTE parser
    state: TerminalState,

    /// The renderer for drawing terminal content
    renderer: TerminalRenderer,

    /// Focus handle for keyboard event handling
    focus_handle: FocusHandle,

    /// Writer for sending input to the terminal process
    stdin_writer: Arc<parking_lot::Mutex<Box<dyn Write + Send>>>,

    /// Receiver for terminal events from the event proxy
    event_rx: mpsc::Receiver<TerminalEvent>,

    /// Configuration used to create this terminal
    config: TerminalConfig,

    /// Async task that reads bytes and notifies the view (push-based)
    #[allow(dead_code)]
    _reader_task: Task<()>,

    /// Callback to notify the PTY about size changes
    resize_callback: Option<Arc<ResizeCallback>>,

    /// Optional callback to intercept key events before terminal processing
    key_handler: Option<Arc<KeyHandler>>,

    /// Callback for terminal bell events
    bell_callback: Option<BellCallback>,

    /// Callback for terminal title changes
    title_callback: Option<TitleCallback>,

    /// Callback for clipboard store requests
    clipboard_store_callback: Option<ClipboardStoreCallback>,

    /// Callback for clipboard load requests
    clipboard_load_callback: Option<ClipboardLoadCallback>,

    /// Callback for terminal exit events
    exit_callback: Option<ExitCallback>,

    /// Scroll accumulator up/down
    scroll_accumulator: f32,

    /// Start selection
    selection_start: Option<alacritty_terminal::index::Point>,

    /// End selection
    selection_end: Option<alacritty_terminal::index::Point>,
}

impl TerminalView {
    /// Create a new terminal with provided I/O streams.
    ///
    /// This method initializes a new terminal emulator with the given stdin writer
    /// and stdout reader. It spawns a background task to read from stdout and
    /// process incoming bytes through the VTE parser.
    ///
    /// # Arguments
    ///
    /// * `stdin_writer` - Writer for sending input bytes to the terminal process
    /// * `stdout_reader` - Reader for receiving output bytes from the terminal process
    /// * `config` - Terminal configuration (dimensions, font, etc.)
    /// * `cx` - GPUI context for this view
    ///
    /// # Returns
    ///
    /// A new `TerminalView` instance ready to be rendered.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // In a GPUI window context:
    /// let terminal = cx.new(|cx| {
    ///     TerminalView::new(stdin_writer, stdout_reader, TerminalConfig::default(), cx)
    /// });
    /// ```
    pub fn new<W, R>(
        stdin_writer: W,
        stdout_reader: R,
        config: TerminalConfig,
        cx: &mut Context<Self>,
    ) -> Self
    where
        W: Write + Send + 'static,
        R: Read + Send + 'static,
    {
        // Create event channel for terminal events
        let (event_tx, event_rx) = mpsc::channel();

        // Clone event_tx for the reader task to send Exit event when PTY closes
        let exit_event_tx = event_tx.clone();

        // Create event proxy for alacritty
        let event_proxy = GpuiEventProxy::new(event_tx);

        // Create terminal state
        let state = TerminalState::new(config.cols, config.rows, event_proxy);

        // Create renderer with font settings and color palette
        let renderer = TerminalRenderer::new(
            config.font_family.clone(),
            config.font_size,
            config.line_height_multiplier,
            config.colors.clone(),
        );

        // Create focus handle
        let focus_handle = cx.focus_handle();

        // Wrap stdin writer in Arc<Mutex> for thread-safe access
        let stdin_writer = Arc::new(parking_lot::Mutex::new(
            Box::new(stdin_writer) as Box<dyn Write + Send>
        ));

        // Create async channel for bytes (push-based notification)
        // Using flume instead of smol::channel because flume is executor-agnostic
        // and properly wakes GPUI's async executor when data arrives
        let (bytes_tx, bytes_rx) = flume::unbounded::<Vec<u8>>();

        // Spawn background thread to read from stdout
        // This thread sends bytes through the async channel
        thread::spawn(move || {
            Self::read_stdout_blocking(stdout_reader, bytes_tx);
        });

        // Spawn async task that awaits on the channel and notifies the view
        // This is push-based: the task blocks until bytes arrive, then immediately notifies
        let reader_task = cx.spawn(async move |this: WeakEntity<Self>, cx: &mut AsyncApp| {
            loop {
                // Wait for bytes from the background reader (blocks until data arrives)
                match bytes_rx.recv_async().await {
                    Ok(bytes) => {
                        // Process bytes and notify the view
                        let result = this.update(cx, |view: &mut Self, cx: &mut Context<Self>| {
                            view.state.process_bytes(&bytes);
                            cx.notify();
                        });
                        if result.is_err() {
                            // View was dropped, exit
                            break;
                        }
                    }
                    Err(_) => {
                        // Channel closed - PTY has finished, send Exit event
                        let _ = exit_event_tx.send(TerminalEvent::Exit);
                        // Notify view to process the Exit event
                        let _ = this.update(cx, |_view, cx: &mut Context<Self>| {
                            cx.notify();
                        });
                        break;
                    }
                }
            }
        });

        Self {
            state,
            renderer,
            focus_handle,
            stdin_writer,
            event_rx,
            config,
            _reader_task: reader_task,
            resize_callback: None,
            key_handler: None,
            bell_callback: None,
            title_callback: None,
            selection_start: None,
            selection_end: None,
            clipboard_store_callback: None,
            clipboard_load_callback: None,
            exit_callback: None,
            scroll_accumulator: 0.,
        }
    }

    /// Set a callback to be invoked when the terminal is resized.
    ///
    /// This callback should resize the underlying PTY to match the new dimensions.
    /// The callback receives (cols, rows) as arguments.
    ///
    /// # Arguments
    ///
    /// * `callback` - A function that will be called with (cols, rows) on resize
    pub fn with_resize_callback(
        mut self,
        callback: impl Fn(usize, usize) + Send + Sync + 'static,
    ) -> Self {
        self.resize_callback = Some(Arc::new(Box::new(callback)));
        self
    }

    /// Set a callback to intercept key events before terminal processing.
    ///
    /// The callback receives the key event and should return `true` to consume
    /// the event (prevent the terminal from processing it), or `false` to allow
    /// normal terminal processing.
    ///
    /// # Arguments
    ///
    /// * `handler` - A function that receives key events and returns whether to consume them
    ///
    /// # Example
    ///
    /// ```ignore
    /// terminal.with_key_handler(|event| {
    ///     // Handle Ctrl++ to increase font size
    ///     if event.keystroke.modifiers.control && event.keystroke.key == "+" {
    ///         // Handle the event
    ///         return true; // Consume the event
    ///     }
    ///     false // Let terminal handle it
    /// })
    /// ```
    pub fn with_key_handler(
        mut self,
        handler: impl Fn(&KeyDownEvent) -> bool + Send + Sync + 'static,
    ) -> Self {
        self.key_handler = Some(Arc::new(Box::new(handler)));
        self
    }

    /// Set a callback to be invoked when the terminal bell is triggered.
    ///
    /// The callback receives a mutable reference to the window and context,
    /// allowing you to play a sound or show a visual indicator.
    ///
    /// # Arguments
    ///
    /// * `callback` - A function that will be called when the bell is triggered
    ///
    /// # Example
    ///
    /// ```ignore
    /// terminal.with_bell_callback(|window, cx| {
    ///     // Play a sound or flash the screen
    /// })
    /// ```
    pub fn with_bell_callback(
        mut self,
        callback: impl Fn(&mut Window, &mut Context<TerminalView>) + 'static,
    ) -> Self {
        self.bell_callback = Some(Box::new(callback));
        self
    }

    /// Set a callback to be invoked when the terminal title changes.
    ///
    /// The callback receives a mutable reference to the window and context,
    /// along with the new title string.
    ///
    /// # Arguments
    ///
    /// * `callback` - A function that will be called with the new title
    ///
    /// # Example
    ///
    /// ```ignore
    /// terminal.with_title_callback(|window, cx, title| {
    ///     // Update window title or tab title
    /// })
    /// ```
    pub fn with_title_callback(
        mut self,
        callback: impl Fn(&mut Window, &mut Context<TerminalView>, &str) + 'static,
    ) -> Self {
        self.title_callback = Some(Box::new(callback));
        self
    }

    /// Set a callback to be invoked when the terminal wants to store data to the clipboard.
    ///
    /// The callback receives a mutable reference to the window and context,
    /// along with the text to store. This is typically triggered by OSC 52 escape sequences.
    ///
    /// # Arguments
    ///
    /// * `callback` - A function that will be called with the text to store
    ///
    /// # Example
    ///
    /// ```ignore
    /// terminal.with_clipboard_store_callback(|window, cx, text| {
    ///     // Store text to system clipboard
    /// })
    /// ```
    pub fn with_clipboard_store_callback(
        mut self,
        callback: impl Fn(&mut Window, &mut Context<TerminalView>, &str) + 'static,
    ) -> Self {
        self.clipboard_store_callback = Some(Box::new(callback));
        self
    }

    /// Set a callback to be invoked when the terminal wants to load data from the clipboard.
    ///
    /// The callback should return the current text in the system clipboard, or `None` to deny the request.
    ///
    /// # Example
    ///
    /// ```ignore
    /// terminal.with_clipboard_load_callback(|window, cx| {
    ///     // Option 1: Use the gpui clipboard
    ///     cx.read_from_clipboard().and_then(|item| item.text())
    ///     
    ///     // Option 2: Always deny for safety
    ///     // None
    /// })
    /// ```
    pub fn with_clipboard_load_callback(
        mut self,
        callback: impl Fn(&mut Window, &mut Context<TerminalView>) -> Option<String> + 'static,
    ) -> Self {
        self.clipboard_load_callback = Some(Box::new(callback));
        self
    }

    /// Set a callback to be invoked when the terminal process exits.
    ///
    /// The callback receives a mutable reference to the window and context,
    /// allowing you to close the terminal view or show an exit message.
    ///
    /// # Arguments
    ///
    /// * `callback` - A function that will be called when the process exits
    ///
    /// # Example
    ///
    /// ```ignore
    /// terminal.with_exit_callback(|window, cx| {
    ///     // Close the terminal tab or show exit message
    /// })
    /// ```
    pub fn with_exit_callback(
        mut self,
        callback: impl Fn(&mut Window, &mut Context<TerminalView>) + 'static,
    ) -> Self {
        self.exit_callback = Some(Box::new(callback));
        self
    }

    /// Background thread that reads from stdout.
    ///
    /// This function runs in a background thread, continuously reading bytes
    /// from the stdout reader and sending them through the async channel.
    /// The async channel allows the main async task to be woken up immediately
    /// when data arrives (push-based).
    fn read_stdout_blocking<R: Read + Send + 'static>(
        mut stdout_reader: R,
        bytes_tx: flume::Sender<Vec<u8>>,
    ) {
        let mut buffer = [0u8; 4096];

        loop {
            match stdout_reader.read(&mut buffer) {
                Ok(0) => {
                    // EOF - channel will be dropped, signaling completion
                    break;
                }
                Ok(n) => {
                    // Send bytes to the async task
                    let bytes = buffer[..n].to_vec();
                    if bytes_tx.send(bytes).is_err() {
                        break; // Channel closed
                    }
                }
                Err(_) => {
                    // Read error
                    break;
                }
            }
        }
    }

    /// Handle keyboard input events.
    ///
    /// Converts GPUI keystrokes to terminal escape sequences and writes them
    /// to the stdin writer. If a key handler is set and returns true, the event
    /// is consumed and not sent to the terminal.
    fn on_key_down(&mut self, event: &KeyDownEvent, window: &mut Window, cx: &mut Context<Self>) {
        let keystroke = &event.keystroke;
        let mode = self.state.mode();

        let is_ctrl_c = keystroke.key == "c" && keystroke.modifiers.control;
        if is_ctrl_c {
            if let (Some(start), Some(end)) = (self.selection_start, self.selection_end)
                && start != end
            {
                let text = self.get_selected_text();
                if !text.is_empty() {
                    self.copy_to_clipboard(text, window, cx);
                    if !keystroke.modifiers.shift {
                        self.selection_start = None;
                        self.selection_end = None;
                        cx.notify();
                        return;
                    }
                }
            }

            if keystroke.modifiers.shift {
                return;
            }
        }

        if keystroke.modifiers.shift
            && matches!(keystroke.key.as_str(), "up" | "down" | "left" | "right")
        {
            self.state.with_term_mut(|term| {
                if self.selection_start.is_none() {
                    let cursor = term.grid().cursor.point;
                    self.selection_start = Some(cursor);
                    self.selection_end = Some(cursor);
                }

                if let Some(mut end) = self.selection_end {
                    match keystroke.key.as_str() {
                        "up" => {
                            let new_line =
                                (end.line.0 - 1).max(-(term.grid().display_offset() as i32));
                            end.line = alacritty_terminal::index::Line(new_line);
                        }
                        "down" => {
                            let new_line =
                                (end.line.0 + 1).min(term.grid().screen_lines() as i32 - 1);
                            end.line = alacritty_terminal::index::Line(new_line);
                        }
                        "left" => {
                            let new_col = (end.column.0 as i32 - 1).max(0) as usize;
                            end.column = alacritty_terminal::index::Column(new_col);
                        }
                        "right" => {
                            let new_col =
                                (end.column.0 as i32 + 1).min(term.columns() as i32 - 1) as usize;
                            end.column = alacritty_terminal::index::Column(new_col);
                        }
                        _ => unreachable!(),
                    }
                    self.selection_end = Some(end);
                }
            });

            cx.notify();
            return;
        }

        if let Some(ref handler) = self.key_handler
            && handler(event)
        {
            return;
        }

        if let Some(bytes) = keystroke_to_bytes(&event.keystroke, mode) {
            let mut writer = self.stdin_writer.lock();
            let _ = writer.write_all(&bytes);
            let _ = writer.flush();

            if self.selection_start.is_some() {
                self.selection_start = None;
                self.selection_end = None;
            }

            self.state.with_term_mut(|term| {
                if term.grid().display_offset() != 0 {
                    term.scroll_display(alacritty_terminal::grid::Scroll::Bottom);
                    self.scroll_accumulator = 0.0;
                }
            });
            cx.notify();
        }
    }

    /// Handle mouse down events.
    ///
    /// Currently a placeholder for future mouse selection and interaction support.
    fn on_mouse_down(
        &mut self,
        event: &MouseDownEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        #[cfg(feature = "gpui")]
        window.focus(&self.focus_handle);

        #[cfg(feature = "gpui-kit")]
        window.focus(&self.focus_handle, cx);

        if event.click_count == 2 {
            self.state.with_term_mut(|term| {
                let point = point_to_cell(
                    self.config.padding,
                    self.renderer.cell_width,
                    self.renderer.cell_height,
                    event.position,
                    term,
                );

                let mut start = point;
                while start.column.0 > 0 {
                    let prev_cell = &term.grid()[alacritty_terminal::index::Point::new(
                        start.line,
                        alacritty_terminal::index::Column(start.column.0 - 1),
                    )];
                    if prev_cell.c.is_whitespace() {
                        break;
                    }
                    start.column.0 -= 1;
                }

                let mut end = point;
                while end.column.0 < term.columns() - 1 {
                    let next_cell = &term.grid()[alacritty_terminal::index::Point::new(
                        end.line,
                        alacritty_terminal::index::Column(end.column.0 + 1),
                    )];
                    if next_cell.c.is_whitespace() {
                        break;
                    }
                    end.column.0 += 1;
                }

                self.selection_start = Some(start);
                self.selection_end = Some(end);
            });
            cx.notify();
            return;
        }

        let mode = self.state.mode();
        if mode.intersects(
            alacritty_terminal::term::TermMode::MOUSE_REPORT_CLICK
                | alacritty_terminal::term::TermMode::MOUSE_MOTION,
        ) {
            self.state.with_term_mut(|term| {
                let point = point_to_cell(
                    self.config.padding,
                    self.renderer.cell_width,
                    self.renderer.cell_height,
                    event.position,
                    term,
                );
                let col = point.column.0 + 1;
                let line = point.line.0 + 1;

                if mode.contains(alacritty_terminal::term::TermMode::SGR_MOUSE) {
                    // \x1b[<0;{col};{line}M  (0 = Left buttom)
                    let seq = format!("\x1b[<0;{};{}M", col, line);
                    let mut writer = self.stdin_writer.lock();
                    let _ = writer.write_all(seq.as_bytes());
                    let _ = writer.flush();
                } else {
                    // Mode X10 legacy
                }
            });
            return;
        }

        self.state.with_term_mut(|term| {
            let point = point_to_cell(
                self.config.padding,
                self.renderer.cell_width,
                self.renderer.cell_height,
                event.position,
                term,
            );
            self.selection_start = Some(point);
            self.selection_end = Some(point);
        });

        cx.notify();
    }

    /// Handle mouse up events.
    ///
    /// Currently a placeholder for future mouse selection support.
    fn on_mouse_up(&mut self, event: &MouseUpEvent, window: &mut Window, cx: &mut Context<Self>) {
        let mode = self.state.mode();

        if mode.intersects(alacritty_terminal::term::TermMode::MOUSE_REPORT_CLICK) {
            self.state.with_term_mut(|term| {
                let point = point_to_cell(
                    self.config.padding,
                    self.renderer.cell_width,
                    self.renderer.cell_height,
                    event.position,
                    term,
                );
                let col = point.column.0 + 1;
                let line = point.line.0 + 1;

                if mode.contains(alacritty_terminal::term::TermMode::SGR_MOUSE) {
                    let seq = format!("\x1b[<0;{};{}m", col, line);
                    let mut writer = self.stdin_writer.lock();
                    let _ = writer.write_all(seq.as_bytes());
                    let _ = writer.flush();
                }
            });
            return;
        }

        if let (Some(start), Some(end)) = (self.selection_start, self.selection_end)
            && start != end
        {
            let selected_text = self.get_selected_text();

            if self.config.copy_on_select && !selected_text.is_empty() {
                self.copy_to_clipboard(selected_text, window, cx);
                self.selection_start = None;
                self.selection_end = None;
            }
        }

        cx.notify();
    }

    /// Handle mouse move events.
    ///
    /// Currently a placeholder for future mouse selection support.
    fn on_mouse_move(
        &mut self,
        event: &MouseMoveEvent,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let mode = self.state.mode();

        if mode.contains(alacritty_terminal::term::TermMode::MOUSE_MOTION)
            && event.pressed_button == Some(MouseButton::Left)
        {
            self.state.with_term_mut(|term| {
                let point = point_to_cell(
                    self.config.padding,
                    self.renderer.cell_width,
                    self.renderer.cell_height,
                    event.position,
                    term,
                );
                let col = point.column.0 + 1;
                let line = point.line.0 + 1;

                if mode.contains(alacritty_terminal::term::TermMode::SGR_MOUSE) {
                    let seq = format!("\x1b[<32;{};{}M", col, line);
                    let mut writer = self.stdin_writer.lock();
                    let _ = writer.write_all(seq.as_bytes());
                    let _ = writer.flush();
                }
            });
            return;
        }

        if self.selection_start.is_some() && event.pressed_button == Some(MouseButton::Left) {
            self.state.with_term_mut(|term| {
                self.selection_end = Some(point_to_cell(
                    self.config.padding,
                    self.renderer.cell_width,
                    self.renderer.cell_height,
                    event.position,
                    term,
                ));
            });
            cx.notify(); // force
        }
    }

    /// Handle scroll events.
    ///
    /// Currently a placeholder for future scrollback support.
    fn on_scroll(&mut self, delta: f32, _window: &mut Window, cx: &mut Context<Self>) {
        // TODO: Implement scrollback
        // - Send scroll reports if alternate screen is not active (missing tests)

        let mode = self.state.mode();

        if mode.contains(alacritty_terminal::term::TermMode::ALT_SCREEN) {
            let lines = delta as i32;
            if lines != 0 {
                let bytes = if lines > 0 {
                    b"\x1b[B".repeat(lines as usize)
                } else {
                    b"\x1b[A".repeat((-lines) as usize)
                };

                let mut writer = self.stdin_writer.lock();
                let _ = writer.write_all(&bytes);
                let _ = writer.flush();
            }
        } else {
            self.scroll_accumulator += delta;
            let lines_to_scroll = self.scroll_accumulator.trunc() as i32;

            if lines_to_scroll != 0 {
                self.scroll_accumulator -= lines_to_scroll as f32;

                self.state.with_term_mut(|term| {
                    term.scroll_display(alacritty_terminal::grid::Scroll::Delta(lines_to_scroll));
                });

                cx.notify();
            }
        }
    }

    fn copy_to_clipboard(&self, text: String, window: &mut Window, cx: &mut Context<Self>) {
        if let Some(ref callback) = self.clipboard_store_callback {
            callback(window, cx, &text);
        }
    }

    fn get_selected_text(&self) -> String {
        let mut text = String::new();
        if let (Some(mut start), Some(mut end)) = (self.selection_start, self.selection_end) {
            if start > end {
                std::mem::swap(&mut start, &mut end);
            }

            self.state.with_term_mut(|term| {
                let grid = term.grid();
                for line_idx in start.line.0..=end.line.0 {
                    let line = alacritty_terminal::index::Line(line_idx);

                    let start_col = if line == start.line {
                        start.column.0
                    } else {
                        0
                    };
                    let end_col = if line == end.line {
                        end.column.0
                    } else {
                        grid.columns() - 1
                    };

                    let mut line_str = String::new();
                    for col_idx in start_col..=end_col {
                        let cell = &grid[alacritty_terminal::index::Point::new(
                            line,
                            alacritty_terminal::index::Column(col_idx),
                        )];
                        if cell.c != '\0' {
                            line_str.push(cell.c);
                        }
                    }

                    text.push_str(line_str.trim_end());

                    // add line
                    if line != end.line {
                        text.push('\n');
                    }
                }
            });
        }
        text
    }

    /// Process pending terminal events.
    ///
    /// This method drains all available events from the event receiver
    /// and handles them appropriately. Note: bytes are processed in the
    /// async reader task, not here.
    fn process_events(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        // Process terminal events (from alacritty event proxy)
        while let Ok(event) = self.event_rx.try_recv() {
            match event {
                TerminalEvent::Wakeup => {
                    // Terminal has new content - already handled by async task
                }
                TerminalEvent::Bell => {
                    if let Some(ref callback) = self.bell_callback {
                        callback(window, cx);
                    }
                }
                TerminalEvent::Title(title) => {
                    if let Some(ref callback) = self.title_callback {
                        callback(window, cx, &title);
                    }
                }
                TerminalEvent::PtyWrite(text) => {
                    let mut writer = self.stdin_writer.lock();
                    let _ = writer.write_all(text.as_bytes());
                    let _ = writer.flush();
                }
                TerminalEvent::ClipboardStore(text) => {
                    if let Some(ref callback) = self.clipboard_store_callback {
                        callback(window, cx, &text);
                    }
                }
                TerminalEvent::ClipboardLoad(formatter) => {
                    if let Some(ref callback) = self.clipboard_load_callback
                        && let Some(text) = callback(window, cx)
                    {
                        let sequence = formatter(&text);
                        let mut writer = self.stdin_writer.lock();
                        let _ = writer.write_all(sequence.as_bytes());
                        let _ = writer.flush();
                    }
                }
                TerminalEvent::Exit => {
                    if let Some(ref callback) = self.exit_callback {
                        callback(window, cx);
                    }
                }
            }
        }
    }

    /// Get the current terminal dimensions.
    ///
    /// # Returns
    ///
    /// A tuple of (columns, rows).
    pub fn dimensions(&self) -> (usize, usize) {
        (self.state.cols(), self.state.rows())
    }

    /// Resize the terminal to new dimensions.
    ///
    /// This method should be called when the terminal view size changes.
    /// It updates the internal grid and notifies the terminal process of the new size.
    ///
    /// # Arguments
    ///
    /// * `cols` - New number of columns
    /// * `rows` - New number of rows
    pub fn resize(&mut self, cols: usize, rows: usize) {
        self.state.resize(cols, rows);
    }

    /// Get the current terminal configuration.
    ///
    /// # Returns
    ///
    /// A reference to the current configuration.
    pub fn config(&self) -> &TerminalConfig {
        &self.config
    }

    /// Get the focus handle for this terminal view.
    ///
    /// # Returns
    ///
    /// A reference to the focus handle.
    pub fn focus_handle(&self) -> &FocusHandle {
        &self.focus_handle
    }

    /// Update the terminal configuration.
    ///
    /// This method updates the terminal's configuration, including font settings,
    /// padding, and color palette. Changes take effect on the next render.
    ///
    /// # Arguments
    ///
    /// * `config` - The new configuration to apply
    /// * `cx` - The context for triggering a repaint
    pub fn update_config(&mut self, config: TerminalConfig, cx: &mut Context<Self>) {
        // Update renderer with new font settings and palette
        self.renderer.font_family = config.font_family.clone();
        self.renderer.font_size = config.font_size;
        self.renderer.line_height_multiplier = config.line_height_multiplier;
        self.renderer.palette = config.colors.clone();

        // Store the new config
        self.config = config;

        // Trigger a repaint - cell dimensions will be recalculated via measure_cell()
        cx.notify();
    }

    /// Calculate terminal dimensions from pixel bounds and cell size.
    ///
    /// Helper method to determine how many columns and rows fit in the given bounds.
    #[allow(dead_code)]
    fn calculate_dimensions(&self, bounds: Bounds<Pixels>) -> (usize, usize) {
        let width_f32: f32 = bounds.size.width.into();
        let height_f32: f32 = bounds.size.height.into();
        let cell_width_f32: f32 = self.renderer.cell_width.into();
        let cell_height_f32: f32 = self.renderer.cell_height.into();

        let cols = ((width_f32 / cell_width_f32) as usize).max(1);
        let rows = ((height_f32 / cell_height_f32) as usize).max(1);
        (cols, rows)
    }
}

impl Render for TerminalView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        self.renderer.measure_cell(window);
        // Process any pending events
        self.process_events(window, cx);

        // Get terminal state and renderer for rendering
        let state_arc = self.state.term_arc();
        let renderer = self.renderer.clone();
        let resize_callback = self.resize_callback.clone();
        let padding = self.config.padding;
        let selection_start = self.selection_start;
        let selection_end = self.selection_end;

        div()
            .size_full()
            .bg(rgb(0x1e1e1e))
            .track_focus(&self.focus_handle)
            .on_key_down(cx.listener(Self::on_key_down))
            .on_mouse_down(MouseButton::Left, cx.listener(Self::on_mouse_down))
            .on_mouse_up(MouseButton::Left, cx.listener(Self::on_mouse_up))
            .on_mouse_move(cx.listener(Self::on_mouse_move))
            .when(true, |el| attach_scroll(el, Self::on_scroll, cx))
            .child(
                canvas(
                    move |bounds, _window, _cx| bounds,
                    move |bounds, _, window, cx| {
                        use alacritty_terminal::grid::Dimensions;

                        // Calculate available space after padding
                        let available_width: f32 =
                            (bounds.size.width - padding.left - padding.right).into();
                        let available_height: f32 =
                            (bounds.size.height - padding.top - padding.bottom).into();
                        let cell_width_f32: f32 = renderer.cell_width.into();
                        let cell_height_f32: f32 = renderer.cell_height.into();

                        let cols = ((available_width / cell_width_f32) as usize).max(1);
                        let rows = ((available_height / cell_height_f32) as usize).max(1);

                        // Helper struct implementing Dimensions for resize
                        struct TermSize {
                            cols: usize,
                            rows: usize,
                        }
                        impl Dimensions for TermSize {
                            fn total_lines(&self) -> usize {
                                self.rows
                            }
                            fn screen_lines(&self) -> usize {
                                self.rows
                            }
                            fn columns(&self) -> usize {
                                self.cols
                            }
                            fn last_column(&self) -> alacritty_terminal::index::Column {
                                alacritty_terminal::index::Column(self.cols.saturating_sub(1))
                            }
                            fn bottommost_line(&self) -> alacritty_terminal::index::Line {
                                alacritty_terminal::index::Line(self.rows as i32 - 1)
                            }
                            fn topmost_line(&self) -> alacritty_terminal::index::Line {
                                alacritty_terminal::index::Line(0)
                            }
                        }

                        // Resize terminal if dimensions changed
                        let mut term = state_arc.lock();
                        let current_cols = term.columns();
                        let current_rows = term.screen_lines();
                        if cols != current_cols || rows != current_rows {
                            // Notify the PTY about the resize
                            if let Some(ref callback) = resize_callback {
                                callback(cols, rows);
                            }
                            term.resize(TermSize { cols, rows });
                        }

                        let selection = match (selection_start, selection_end) {
                            (Some(start), Some(end)) => Some((start, end)),
                            _ => None,
                        };

                        // Paint the terminal with measured dimensions
                        renderer.paint(bounds, padding, &*term, selection, window, cx);
                    },
                )
                .size_full(),
            )
    }
}

// Tests are omitted due to macro expansion issues with the test attribute
// in this configuration. Integration tests can be added separately.

fn point_to_cell(
    paddings: Edges<Pixels>,
    cell_width: Pixels,
    cell_height: Pixels,
    pos: Point<Pixels>,
    term: &Term<GpuiEventProxy>,
) -> alacritty_terminal::index::Point {
    let x = pos.x - paddings.left;
    let y = pos.y - paddings.top;

    let col = (f32::from(x) / f32::from(cell_width)).floor().max(0.0) as usize;
    let line = (f32::from(y) / f32::from(cell_height)).floor().max(0.0) as usize;

    let col = col.min(term.columns().saturating_sub(1));
    let line = line.min(term.screen_lines().saturating_sub(1));

    let line_idx =
        alacritty_terminal::index::Line(line as i32 - term.grid().display_offset() as i32);
    let col_idx = alacritty_terminal::index::Column(col);

    alacritty_terminal::index::Point::new(line_idx, col_idx)
}
