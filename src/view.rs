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
pub use alacritty_terminal::vte::ansi::CursorShape;

use crate::{GridSource, mouse};
// use crate::GridSource;
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
use std::time::Duration;

/// Configuration for terminal creation and runtime updates.
///
/// This struct defines the terminal's appearance and behavior, including
/// grid dimensions, font settings, scrollback buffer, color scheme, and
/// cursor style.
///
/// # Default Values
///
/// | Field | Default |
/// |-------|---------|
/// | `cols` | 80 |
/// | `rows` | 24 |
/// | `font_family` | `"monospace"` |
/// | `font_size` | 14px |
/// | `scrollback` | 10000 |
/// | `line_height_multiplier` | 1.0 |
/// | `padding` | 0px all sides |
/// | `colors` | Default palette |
/// | `copy_on_select` | false |
/// | `cursor_blink_override` | `None` (respect the program) |
/// | `cursor_blink_ms` | 500 |
/// | `cursor_thickness` | 0.15 |
/// | `cursor_shape_override` | `None` (respect the program) |
///
/// # Example
///
/// ```ignore
/// use alacritty_terminal::vte::ansi::CursorShape;
/// use gpui::{Edges, px};
/// use gpui_terminal::{ColorPalette, TerminalConfig};
///
/// let config = TerminalConfig {
///     cols: 120,
///     rows: 40,
///     font_family: "JetBrains Mono".into(),
///     font_size: px(13.0),
///     scrollback: 50_000,
///     line_height_multiplier: 1.0,
///     padding: Edges::all(px(10.0)),
///     colors: ColorPalette::builder()
///         .background(0x1a, 0x1a, 0x1a)
///         .foreground(0xe0, 0xe0, 0xe0)
///         .build(),
///     copy_on_select: true,
///     cursor_blink_override: Some(true),
///     cursor_blink_ms: 500,
///     cursor_thickness: 0.15,
///     cursor_shape_override: Some(CursorShape::Beam),
/// };
/// ```
///
/// # Cursor Overrides
///
/// Programs can request a cursor shape and blink state through the
/// `DECSCUSR` escape sequence (`CSI Ps SP q`). The `*_override` fields
/// let the host application ignore those requests and enforce a style
/// of its own:
///
/// - `Some(value)` → always use this value, regardless of what the
///   program requests.
/// - `None` → honor the grid's cursor state (see
///   [`GridSource::cursor`](crate::grid::GridSource::cursor)).
///
/// `cursor_thickness` only affects `Beam` and `Underline` cursors: a
/// `Block` cursor fills the entire cell by definition. The value is a
/// fraction of the cell's width (for `Beam`) or height (for
/// `Underline`), so `0.15` means 15%.
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

    /// Multiplier for line height to accommodate tall glyphs (e.g., nerd fonts).
    /// Default is 1.0 (no extra height).
    pub line_height_multiplier: f32,

    /// Padding around the terminal content (top, right, bottom, left).
    /// The padding area renders with the terminal's background color.
    pub padding: Edges<Pixels>,

    /// Color palette for terminal colors (16 ANSI colors, 256 extended colors,
    /// foreground, background, and cursor colors)
    pub colors: ColorPalette,

    /// Copy to clipboard on text selection
    pub copy_on_select: bool,

    /// Forces the cursor to blink (or not), ignoring what the program
    /// requests via `DECSCUSR`.
    ///
    /// - `None` → honor the cursor state reported by
    ///   [`GridSource::cursor`](crate::grid::GridSource::cursor).
    /// - `Some(true)` → always blink.
    /// - `Some(false)` → never blink.
    pub cursor_blink_override: Option<bool>,

    /// Interval, in milliseconds, between cursor blink phases.
    /// Ignored when the effective blink state is `false`.
    pub cursor_blink_ms: u64,

    /// Thickness of `Beam` and `Underline` cursors as a fraction of the
    /// cell's width or height respectively (e.g. `0.15` = 15%).
    ///
    /// Has no effect on `Block` cursors, which fill the entire cell.
    pub cursor_thickness: f32,

    /// Forces the cursor shape, ignoring what the program requests via
    /// `DECSCUSR`.
    ///
    /// - `None` → honor the shape reported by
    ///   [`GridSource::cursor`](crate::grid::GridSource::cursor).
    /// - `Some(shape)` → always draw this shape (`Block`, `Beam`,
    ///   `Underline`, …).
    pub cursor_shape_override: Option<CursorShape>,
}

/// Provides the documented defaults listed on [`TerminalConfig`].
///
/// Notably, both cursor overrides default to `None`, so programs that emit
/// `DECSCUSR` keep full control of the cursor unless the host application
/// opts in by setting an override.
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
            cursor_blink_override: None,
            cursor_blink_ms: 500,
            cursor_thickness: 0.15,
            cursor_shape_override: None,
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
/// providing a complete terminal emulator that can be embedded in any GPUI
/// application.
///
/// # Responsibilities
///
/// - **Terminal State**: Manages the grid, cursor, and colors via
///   [`TerminalState`].
/// - **I/O Streams**: Reads from PTY stdout and writes to PTY stdin.
/// - **Event Handling**: Processes keyboard, mouse, and resize events.
/// - **Rendering**: Paints text, backgrounds, and cursor via
///   [`TerminalRenderer`].
/// - **Cursor Animation**: Drives the cursor blink timer when the effective
///   cursor style requests it (see [Cursor Blinking](#cursor-blinking)).
/// - **Callbacks**: Dispatches events to user-provided callbacks.
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
/// - [`with_clipboard_store_callback`](Self::with_clipboard_store_callback) -
///   Clipboard writes
///
/// # Cursor Blinking
///
/// Blinking is driven by a per-view timer rather than the renderer, because
/// it needs to toggle state and repaint the view on an interval. The view
/// keeps three pieces of state for this:
///
/// - `cursor_visible`: the current phase (`true` = cursor painted this frame).
/// - `blink_task`: the background task currently running the blink loop, if
///   any.
/// - `blink_epoch`: a generation counter. Every time the loop is restarted
///   (e.g. because the user typed, or the effective blink state changed) the
///   epoch is bumped, and any in-flight task whose epoch no longer matches
///   exits on its next tick. This avoids the classic "task drops itself from
///   inside itself" bug and makes cancellation race-free.
///
/// The **effective** blink state is resolved as:
///
/// ```text
/// config.cursor_blink_override.unwrap_or(grid.cursor().blink)
/// ```
///
/// so a host application can force blinking on or off regardless of what the
/// program requests via `DECSCUSR`. The same resolved state is what gets passed
/// to [`TerminalRenderer::paint`], ensuring the renderer's hide/show decision
/// matches the timer's phase. Reading `grid.cursor().blink` directly in
/// `paint` would ignore the override and break blinking.
///
/// # Thread Safety
///
/// `TerminalView` is not `Send` as it contains GPUI handles. The stdin writer
/// is internally wrapped in `Arc<parking_lot::Mutex<>>` for safe concurrent
/// access.
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

    /// Configuration used to create this terminal.
    ///
    /// Mutated through [`update_config`](Self::update_config); the cursor
    /// `*_override` fields are re-read on every render, so changing them takes
    /// effect immediately without restarting the blink task.
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

    /// Fractional scroll offset accumulated from wheel/trackpad events.
    ///
    /// Wheel deltas arrive in fractional units; whole lines are extracted from
    /// this accumulator and applied to the grid's `display_offset`, while the
    /// remainder carries over to the next event. This keeps scrolling smooth
    /// on high-resolution trackpads, where a single "notch" may span several
    /// frames.
    scroll_accumulator: f32,

    /// Current text selection, or `None` when nothing is selected.
    ///
    /// Replaces the previous `selection_start` / `selection_end` pair.
    /// The [`SelectionType`] distinguishes single-click (Simple),
    /// double-click (Word), and triple-click (Line) selections, and is
    /// preserved across drags. Endpoints are stored in buffer coordinates
    /// (negative lines reach into the scrollback).
    selection: Option<mouse::Selection>,

    /// Current phase of the cursor blink cycle.
    ///
    /// `true` means the cursor is painted this frame. Toggled by
    /// [`blink_task`](Self::blink_task) when the effective cursor style
    /// requests blinking; forced back to `true` whenever the user types, so
    /// the cursor is always visible while input is happening. Passed to
    /// [`TerminalRenderer::paint`] to gate the actual draw.
    cursor_visible: bool,

    /// Background task driving the cursor blink loop, if one is running.
    ///
    /// Set by [`sync_blink`](Self::sync_blink) when the effective cursor
    /// style requests blinking; left as `None` otherwise. The task never
    /// clears this field from within itself — instead it checks
    /// [`blink_epoch`](Self::blink_epoch) on every tick and exits if it has
    /// been superseded. External code (key handlers, `sync_blink`) sets this
    /// to `None` to cancel the loop, after bumping the epoch.
    blink_task: Option<Task<()>>,

    /// Generation counter for the cursor blink loop.
    ///
    /// Incremented every time [`sync_blink`](Self::sync_blink) starts a new
    /// loop (or cancels the current one). A running task captures the epoch
    /// it was spawned with and exits on its next tick if `blink_epoch` no
    /// longer matches. This makes restarts race-free without needing to drop
    /// the task from inside itself.
    blink_epoch: usize,
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
            config.cursor_thickness,
            config.colors.clone(),
            config.cursor_shape_override,
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
            selection: None,
            clipboard_store_callback: None,
            clipboard_load_callback: None,
            exit_callback: None,
            scroll_accumulator: 0.0,
            cursor_visible: true,
            blink_task: None,
            blink_epoch: 0,
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
    ///
    /// Reads in 4 KiB chunks. Returns (and therefore closes the channel) on
    /// EOF, on a read error, or when the receiver has been dropped — the async
    /// side treats the channel closing as "PTY finished" and emits
    /// [`TerminalEvent::Exit`].
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
    ///
    /// # Dispatch order
    ///
    /// 1. **Ctrl+C with a selection**: copies the selection to the clipboard
    ///    and, unless Shift is held, clears it and swallows the event.
    /// 2. **Shift+Arrow keys**: extend the current selection instead of
    ///    sending cursor-movement sequences to the PTY.
    /// 3. **`key_handler` callback**: if it returns `true`, the event is
    ///    considered consumed and no bytes are sent.
    /// 4. **Default path**: translate the keystroke to bytes via
    ///    [`keystroke_to_bytes`], write them to the PTY, drop any existing
    ///    selection, scroll the view back to the bottom, and restart the
    ///    blink cycle so the cursor is visible while typing.
    fn on_key_down(&mut self, event: &KeyDownEvent, window: &mut Window, cx: &mut Context<Self>) {
        let keystroke = &event.keystroke;
        let mode = self.state.mode();

        let is_ctrl_c = keystroke.key == "c" && keystroke.modifiers.control;
        if is_ctrl_c {
            if let Some(sel) = self.selection.clone()
                && sel.start != sel.end
            {
                let text = self.get_selected_text();
                if !text.is_empty() {
                    self.copy_to_clipboard(text, window, cx);
                    if !keystroke.modifiers.shift {
                        self.selection = None;
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
                if self.selection.is_none() {
                    let cursor = term.grid().cursor.point;
                    self.selection = Some(mouse::Selection::new(
                        cursor,
                        cursor,
                        mouse::SelectionType::Simple,
                    ));
                }

                if let Some(sel) = self.selection.as_mut() {
                    let mut end = sel.end;
                    match keystroke.key.as_str() {
                        "up" => {
                            let new_line = (end.line.0 - 1).max(-(term.display_offset() as i32));
                            end.line = alacritty_terminal::index::Line(new_line);
                        }
                        "down" => {
                            let new_line = (end.line.0 + 1).min(term.screen_lines() as i32 - 1);
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
                    sel.end = end;
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

            if self.selection.is_some() {
                self.selection = None;
            }

            self.state.with_term_mut(|term| {
                if term.grid().display_offset() != 0 {
                    term.scroll_display(alacritty_terminal::grid::Scroll::Bottom);
                    self.scroll_accumulator = 0.0;
                }
            });

            self.cursor_visible = true;
            self.blink_task = None;
            self.blink_epoch += 1;

            cx.notify();
        }
    }

    /// Handle mouse-down events.
    ///
    /// Behavior depends on the terminal mode reported by the program:
    ///
    /// - **Double-click** (`click_count == 2`): select the word under the
    ///   cursor by expanding left/right until whitespace is hit.
    /// - **Mouse reporting active** (`MOUSE_REPORT_CLICK` or `MOUSE_MOTION`):
    ///   forward the event to the PTY as a `SGR` or legacy `X10` mouse
    ///   report, so programs like vim, tmux, or htop can handle it.
    /// - **Otherwise**: start a new text selection at the clicked cell.
    ///
    /// The terminal is also focused by this handler so subsequent key events
    /// are delivered to it.
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

        let selection_type = mouse::selection_type_from_clicks(event.click_count);
        let mode = self.state.mode();

        if selection_type == mouse::SelectionType::Line {
            self.state.with_term_mut(|term| {
                let point = point_to_cell(
                    self.config.padding,
                    self.renderer.cell_width,
                    self.renderer.cell_height,
                    event.position,
                    term,
                );
                let start = alacritty_terminal::index::Point::new(
                    point.line,
                    alacritty_terminal::index::Column(0),
                );
                let end = alacritty_terminal::index::Point::new(
                    point.line,
                    alacritty_terminal::index::Column(term.columns().saturating_sub(1)),
                );
                self.selection = Some(mouse::Selection::new(
                    start,
                    end,
                    mouse::SelectionType::Line,
                ));
            });
            cx.notify();
            return;
        }

        if selection_type == mouse::SelectionType::Word {
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
                    let prev = &term.cell(start.line.0 as usize, start.column.0 - 1);
                    if prev.c.is_whitespace() {
                        break;
                    }
                    start.column.0 -= 1;
                }

                let mut end = point;
                while end.column.0 < term.columns() - 1 {
                    let next = term.cell(end.line.0 as usize, end.column.0 + 1);
                    if next.c.is_whitespace() {
                        break;
                    }
                    end.column.0 += 1;
                }

                self.selection = Some(mouse::Selection::new(
                    start,
                    end,
                    mouse::SelectionType::Word,
                ));
            });
            cx.notify();
            return;
        }

        if mode.intersects(
            alacritty_terminal::term::TermMode::MOUSE_REPORT_CLICK
                | alacritty_terminal::term::TermMode::MOUSE_MOTION,
        ) && mode.contains(alacritty_terminal::term::TermMode::SGR_MOUSE)
        {
            self.state.with_term_mut(|term| {
                let buffer_point = point_to_cell(
                    self.config.padding,
                    self.renderer.cell_width,
                    self.renderer.cell_height,
                    event.position,
                    term,
                );
                // Mouse reports use *screen* coordinates (0-based from the top
                // of the visible area), not buffer coordinates.
                let screen_line = buffer_point.line.0 + term.grid().display_offset() as i32;
                let screen_point = alacritty_terminal::index::Point::new(
                    alacritty_terminal::index::Line(screen_line),
                    buffer_point.column,
                );

                let modifiers = mouse::encode_modifiers(
                    event.modifiers.shift,
                    event.modifiers.alt,
                    event.modifiers.control,
                );
                if let Some(bytes) = mouse::mouse_button_report(
                    MouseButton::Left,
                    true,
                    screen_point,
                    modifiers,
                    mode,
                ) {
                    let mut writer = self.stdin_writer.lock();
                    let _ = writer.write_all(&bytes);
                    let _ = writer.flush();
                }
            });
            return;
        }

        // Simple selection: start at the clicked cell.
        self.state.with_term_mut(|term| {
            let point = point_to_cell(
                self.config.padding,
                self.renderer.cell_width,
                self.renderer.cell_height,
                event.position,
                term,
            );
            self.selection = Some(mouse::Selection::new(
                point,
                point,
                mouse::SelectionType::Simple,
            ));
        });
        cx.notify();
    }

    /// Handle mouse-up events.
    ///
    /// Two paths, mirroring [`Self::on_mouse_down`]:
    ///
    /// - If the program enabled mouse reporting (`MOUSE_REPORT_CLICK`), the
    ///   release is forwarded as an `SGR` mouse report (`…m` instead of `…M`).
    /// - Otherwise, if a selection was in progress and `copy_on_select` is
    ///   enabled in the config, the selected text is copied to the clipboard
    ///   and the selection is cleared.
    fn on_mouse_up(&mut self, event: &MouseUpEvent, window: &mut Window, cx: &mut Context<Self>) {
        let mode = self.state.mode();

        if mode.contains(alacritty_terminal::term::TermMode::MOUSE_REPORT_CLICK)
            && mode.contains(alacritty_terminal::term::TermMode::SGR_MOUSE)
        {
            self.state.with_term_mut(|term| {
                let buffer_point = point_to_cell(
                    self.config.padding,
                    self.renderer.cell_width,
                    self.renderer.cell_height,
                    event.position,
                    term,
                );
                let screen_line = buffer_point.line.0 + term.grid().display_offset() as i32;
                let screen_point = alacritty_terminal::index::Point::new(
                    alacritty_terminal::index::Line(screen_line),
                    buffer_point.column,
                );

                let modifiers = mouse::encode_modifiers(
                    event.modifiers.shift,
                    event.modifiers.alt,
                    event.modifiers.control,
                );
                if let Some(bytes) = mouse::mouse_button_report(
                    MouseButton::Left,
                    false,
                    screen_point,
                    modifiers,
                    mode,
                ) {
                    let mut writer = self.stdin_writer.lock();
                    let _ = writer.write_all(&bytes);
                    let _ = writer.flush();
                }
            });
            return;
        }

        if let Some(sel) = &self.selection
            && sel.start != sel.end
        {
            let selected_text = self.get_selected_text();
            if self.config.copy_on_select && !selected_text.is_empty() {
                self.copy_to_clipboard(selected_text, window, cx);
                self.selection = None;
            }
        }

        cx.notify();
    }

    /// Handle mouse-move events.
    ///
    /// Two paths:
    ///
    /// - If the program enabled **mouse motion reporting** and the left button
    ///   is held, the drag is forwarded as an `SGR` drag report
    ///   (`\x1b[<32;{col};{line}M`).
    /// - Otherwise, if a selection was started with [`Self::on_mouse_down`],
    ///   the selection end is updated to the cell under the pointer, so the
    ///   renderer redraws the highlight on the next frame.
    fn on_mouse_move(
        &mut self,
        event: &MouseMoveEvent,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let mode = self.state.mode();

        if mode.contains(alacritty_terminal::term::TermMode::MOUSE_MOTION)
            && event.pressed_button == Some(MouseButton::Left)
            && mode.contains(alacritty_terminal::term::TermMode::SGR_MOUSE)
        {
            self.state.with_term_mut(|term| {
                let buffer_point = point_to_cell(
                    self.config.padding,
                    self.renderer.cell_width,
                    self.renderer.cell_height,
                    event.position,
                    term,
                );
                let screen_line = buffer_point.line.0 + term.grid().display_offset() as i32;
                let screen_point = alacritty_terminal::index::Point::new(
                    alacritty_terminal::index::Line(screen_line),
                    buffer_point.column,
                );

                let modifiers = mouse::encode_modifiers(
                    event.modifiers.shift,
                    event.modifiers.alt,
                    event.modifiers.control,
                );
                // Motion report: button bits are 32 (no button) | modifiers.
                let button_value = 32u8 | modifiers;
                let seq = format!(
                    "\x1b[<{};{};{}M",
                    button_value,
                    screen_point.column.0 + 1,
                    screen_point.line.0 + 1
                );
                let mut writer = self.stdin_writer.lock();
                let _ = writer.write_all(seq.as_bytes());
                let _ = writer.flush();
            });
            return;
        }

        if self.selection.is_some() && event.pressed_button == Some(MouseButton::Left) {
            self.state.with_term_mut(|term| {
                let point = point_to_cell(
                    self.config.padding,
                    self.renderer.cell_width,
                    self.renderer.cell_height,
                    event.position,
                    term,
                );
                if let Some(sel) = self.selection.as_mut() {
                    if sel.selection_type == mouse::SelectionType::Line {
                        // Line selection: keep full lines between anchor and pointer.
                        let start_line = sel.start.line.min(point.line);
                        let end_line = sel.start.line.max(point.line);
                        sel.start = alacritty_terminal::index::Point::new(
                            start_line,
                            alacritty_terminal::index::Column(0),
                        );
                        sel.end = alacritty_terminal::index::Point::new(
                            end_line,
                            alacritty_terminal::index::Column(term.columns().saturating_sub(1)),
                        );
                    } else {
                        sel.end = point;
                    }
                }
            });
            cx.notify();
        }
    }

    /// Handle scroll events.
    ///
    /// Deltas are accumulated in [`scroll_accumulator`](Self::scroll_accumulator)
    /// so fractional trackpad deltas translate into whole lines. Whole lines
    /// are then dispatched via [`scroll_report`](crate::mouse::scroll_report):
    ///
    /// - **Mouse mode**: SGR wheel report (`button 64` up / `65` down).
    /// - **Alt-screen without mouse**: `Up`/`Down` arrow keys (with the
    ///   correct `APP_CURSOR` variant).
    /// - **Normal screen without mouse**: `scroll_report` returns `None`, and
    ///   we scroll the local display via `Term::scroll_display` (scrollback).
    fn on_scroll(&mut self, delta: f32, _window: &mut Window, cx: &mut Context<Self>) {
        let mode = self.state.mode();

        self.scroll_accumulator += delta;
        let lines_to_scroll = self.scroll_accumulator.trunc() as i32;
        if lines_to_scroll == 0 {
            return;
        }
        self.scroll_accumulator -= lines_to_scroll as f32;

        // Try to report to the program (mouse mode or alt-screen).
        if let Some(bytes) = mouse::scroll_report(
            lines_to_scroll,
            alacritty_terminal::index::Point::new(
                alacritty_terminal::index::Line(0),
                alacritty_terminal::index::Column(0),
            ),
            0,
            mode,
        ) {
            let mut writer = self.stdin_writer.lock();
            let _ = writer.write_all(&bytes);
            let _ = writer.flush();
            return;
        }

        // Otherwise, scroll the local display (scrollback).
        self.state.with_term_mut(|term| {
            term.scroll_display(alacritty_terminal::grid::Scroll::Delta(lines_to_scroll));
        });
        cx.notify();
    }
    /// Copy text to the system clipboard.
    ///
    /// Dispatches to the user-provided
    /// [`clipboard_store_callback`](Self::with_clipboard_store_callback). If
    /// no callback is set, this is a no-op: the crate does not depend on a
    /// specific clipboard implementation.
    fn copy_to_clipboard(&self, text: String, window: &mut Window, cx: &mut Context<Self>) {
        if let Some(ref callback) = self.clipboard_store_callback {
            callback(window, cx, &text);
        }
    }

    /// Start or stop the cursor blink loop to match `blink`.
    ///
    /// `blink` is the **effective** blink state, already resolved through
    /// [`TerminalConfig::cursor_blink_override`] in [`Render::render`]; this
    /// function never reads the grid directly.
    ///
    /// # Cancellation model
    ///
    /// This function is the *only* place that is allowed to set
    /// [`blink_task`](Self::blink_task) to `None` (besides the key handler).
    /// The task itself never clears the field: it checks
    /// [`blink_epoch`](Self::blink_epoch) on every tick and exits when its
    /// captured epoch no longer matches the current one. That is what makes
    /// restarts race-free and avoids dropping the task from inside itself.
    ///
    /// # Transitions
    ///
    /// - `false → any`: if a task was running, bump the epoch (invalidating it
    ///   on its next tick), clear [`blink_task`](Self::blink_task), and force
    ///   [`cursor_visible`](Self::cursor_visible) back to `true` so a
    ///   non-blinking cursor is always shown.
    /// - `true → already running`: no-op; the existing loop keeps ticking.
    /// - `true → not running`: spawn a new loop that toggles
    ///   [`cursor_visible`](Self::cursor_visible) every
    ///   [`TerminalConfig::cursor_blink_ms`] milliseconds and calls
    ///   `cx.notify()` so the view repaints.
    fn sync_blink(&mut self, blink: bool, cx: &mut Context<Self>) {
        if !blink {
            if self.blink_task.is_some() {
                self.blink_epoch += 1;
                self.blink_task = None;
                self.cursor_visible = true;
                cx.notify();
            }
            return;
        }

        if self.blink_task.is_some() {
            return;
        }

        self.blink_epoch += 1;
        let epoch = self.blink_epoch;
        let interval = self.config.cursor_blink_ms;

        self.blink_task = Some(cx.spawn(async move |this, cx| {
            loop {
                cx.background_executor()
                    .timer(Duration::from_millis(interval))
                    .await;

                let alive = this
                    .update(cx, |this, cx| {
                        if this.blink_epoch != epoch {
                            return false;
                        }
                        this.cursor_visible = !this.cursor_visible;
                        cx.notify();
                        true
                    })
                    .unwrap_or(false);

                if !alive {
                    break;
                }
            }
        }));
    }

    /// Extract the current selection as a plain string.
    ///
    /// The endpoints are normalized (swapped if needed so `start <= end`),
    /// then each buffer line in the range is copied cell by cell. Trailing
    /// whitespace per line is trimmed and lines are joined with `\n`.
    ///
    /// Selection endpoints are stored in **buffer** coordinates (negative
    /// lines reach into the scrollback), but [`GridSource::cell`] expects
    /// **visible** rows, so each iteration adds the display offset back:
    /// `visible = buffer + offset`. Without this conversion, selections that
    /// reach into scrollback would index with a wrapped `usize` and either
    /// panic or return the wrong cell.
    fn get_selected_text(&self) -> String {
        let mut text = String::new();
        if let Some(sel) = &self.selection {
            let (mut start, mut end) = (sel.start, sel.end);
            if start > end {
                std::mem::swap(&mut start, &mut end);
            }

            self.state.with_term_mut(|term| {
                let offset = term.grid().display_offset() as i32;
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
                        term.columns() - 1
                    };

                    let mut line_str = String::new();
                    // Convert buffer line → visible line before querying.
                    let visible_line = (line_idx + offset) as usize;
                    for col_idx in start_col..=end_col {
                        let cell = term.cell(visible_line, col_idx);
                        if cell.c != '\0' {
                            line_str.push(cell.c);
                        }
                    }

                    text.push_str(line_str.trim_end());
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
    ///
    /// Handled events:
    ///
    /// - `Wakeup`: already covered by the reader task, no-op here.
    /// - `Bell`, `Title`, `ClipboardStore`, `ClipboardLoad`, `Exit`: dispatched
    ///   to the corresponding user-provided callback if one was configured.
    /// - `PtyWrite`: written straight to the PTY stdin (used by alacritty for
    ///   responses to terminal queries, e.g. DA, DS R).
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
    /// The renderer's cached font settings (`font_family`, `font_size`,
    /// `line_height_multiplier`, `cursor_thickness`, `palette`,
    /// `cursor_shape_override`) are updated eagerly; cell metrics are
    /// recalculated on the next frame via `TerminalRenderer::measure_cell`.
    ///
    /// Cursor blink settings (`cursor_blink_override`, `cursor_blink_ms`) are
    /// **not** applied here — they are read from `self.config` on every render,
    /// so updating them takes effect on the next frame without restarting the
    /// blink task.
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
    /// Uses the currently measured cell size (from `TerminalRenderer`); callers
    /// should ensure `measure_cell` has run at least once (the first `render`
    /// call does).
    ///
    /// # Arguments
    ///
    /// * `bounds` - Pixel bounds to fit the grid into
    ///
    /// # Returns
    ///
    /// `(cols, rows)`, both clamped to at least 1 so the grid never becomes
    /// degenerate.
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

/// Renders the terminal into the GPUI element tree.
///
/// Each frame does the following, in order:
///
/// 1. Re-measures the cell using [`TerminalRenderer::measure_cell`] (cheap;
///    recalculates font metrics if the font changed since last frame).
/// 2. Drains pending [`TerminalEvent`]s via
///    [`process_events`](TerminalView::process_events) so callbacks fire in
///    the same frame as the event.
/// 3. Reads the grid's blink state, resolves it against
///    [`TerminalConfig::cursor_blink_override`], and calls
///    [`sync_blink`](TerminalView::sync_blink) with the result. This is the
///    single point where the timer is started/stopped.
/// 4. Builds the outer `div` (focus, keyboard, mouse handlers, scroll), and
///    inside it a full-size `canvas` whose paint closure:
///    - computes `cols`/`rows` from the current bounds and cell size,
///    - resizes the `Term` and calls the PTY resize callback if they changed,
///    - resolves the selection endpoints,
///    - calls [`TerminalRenderer::paint`] passing the resolved blink state
///      (so `paint`'s hide/show decision matches the timer's phase).
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
        let selection = self.selection.as_ref().map(|s| (s.start, s.end));
        let cursor_visible = self.cursor_visible;

        let blink_from_grid = self.state.with_term(|term| match term.cursor() {
            Some(c) => c.blink,
            None => false,
        });

        let blink = self.config.cursor_blink_override.unwrap_or(blink_from_grid);

        self.sync_blink(blink, cx);

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
                        // Calculate available space after padding
                        let available_width: f32 =
                            (bounds.size.width - padding.left - padding.right).into();
                        let available_height: f32 =
                            (bounds.size.height - padding.top - padding.bottom).into();
                        let cell_width_f32: f32 = renderer.cell_width.into();
                        let cell_height_f32: f32 = renderer.cell_height.into();

                        let cols = ((available_width / cell_width_f32) as usize).max(1);
                        let rows = ((available_height / cell_height_f32) as usize).max(1);

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

                        // Paint the terminal with measured dimensions
                        renderer.paint(
                            bounds,
                            padding,
                            &*term,
                            selection,
                            cursor_visible,
                            blink,
                            window,
                            cx,
                        );
                    },
                )
                .size_full(),
            )
    }
}

// Tests are omitted due to macro expansion issues with the test attribute
// in this configuration. Integration tests can be added separately.

/// Translate a window-space position into a grid cell.
///
/// # Coordinate system
///
/// `pos` is in window pixels. The function subtracts `paddings` so the top-left
/// of the content area maps to `(0, 0)`, then divides by the cell dimensions
/// and floors to get the cell index. Negative results are clamped to 0 and
/// coordinates past the last column/row are clamped to the grid bounds, so a
/// click just outside the content area still resolves to the nearest cell.
///
/// # Scrollback
///
/// The visible row index is translated into alacritty's buffer coordinates by
/// subtracting `display_offset`. This is what makes clicking while scrolled
/// back select the row the user actually sees, rather than the live row at the
/// bottom of the buffer.
///
/// # Arguments
///
/// * `paddings` - Content padding from [`TerminalConfig::padding`]
/// * `cell_width` / `cell_height` - Measured cell size from the renderer
/// * `pos` - Pointer position in window pixels
/// * `term` - The terminal whose dimensions and scroll offset are used for
///   clamping and translation
///
/// # Returns
///
/// A grid point in buffer coordinates (may be negative to refer to scrollback
/// rows).
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

/// Adapter that exposes a target `(cols, rows)` pair as alacritty's
/// [`Dimensions`](alacritty_terminal::grid::Dimensions).
///
/// Used only to feed `Term::resize` when the canvas size changes. The terminal
/// itself is the source of truth for the current size; this type is a
/// transient "desired size" description.
struct TermSize {
    /// Desired number of columns.
    cols: usize,
    /// Desired number of rows.
    rows: usize,
}

/// [`Dimensions`](alacritty_terminal::grid::Dimensions) adapter for
/// [`TermSize`].
///
/// All methods read directly from the `cols`/`rows` fields; there is no
/// scrollback in the adapter, so `total_lines == screen_lines == rows` and
/// `topmost_line == Line(0)`. `last_column` and `bottommost_line` saturate at
/// 0 for empty dimensions, which matches alacritty's own grid behavior and
/// avoids underflow on a fully collapsed canvas.
impl alacritty_terminal::grid::Dimensions for TermSize {
    /// Total number of lines including scrollback. Equal to `screen_lines`
    /// here because the adapter has no history.
    fn total_lines(&self) -> usize {
        self.rows
    }
    /// Number of visible lines.
    fn screen_lines(&self) -> usize {
        self.rows
    }
    /// Number of columns.
    fn columns(&self) -> usize {
        self.cols
    }
    /// Index of the last column, saturating to 0 for empty grids.
    fn last_column(&self) -> alacritty_terminal::index::Column {
        alacritty_terminal::index::Column(self.cols.saturating_sub(1))
    }
    /// Index of the bottommost visible line.
    fn bottommost_line(&self) -> alacritty_terminal::index::Line {
        alacritty_terminal::index::Line(self.rows as i32 - 1)
    }
    /// Index of the topmost line; always `Line(0)` (no scrollback).
    fn topmost_line(&self) -> alacritty_terminal::index::Line {
        alacritty_terminal::index::Line(0)
    }
}
