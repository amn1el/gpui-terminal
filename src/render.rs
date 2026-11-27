//! Terminal rendering module.
//!
//! This module provides [`TerminalRenderer`], which handles efficient rendering of
//! terminal content using GPUI's text and drawing systems.
//!
//! # Rendering Pipeline
//!
//! The renderer processes the terminal grid in several stages:
//!
//! ```text
//! Terminal Grid → Layout Phase → Paint Phase
//!                      │              │
//!                      ├─ Collect backgrounds
//!                      ├─ Batch text runs
//!                      │              │
//!                      │              ├─ Paint default background
//!                      │              ├─ Paint non-default backgrounds
//!                      │              ├─ Paint text characters
//!                      │              └─ Paint cursor
//! ```
//!
//! # Optimizations
//!
//! The renderer includes several optimizations to minimize draw calls:
//!
//! 1. **Background Merging**: Adjacent cells with the same background color are
//!    merged into single rectangles, reducing the number of quads to paint.
//!
//! 2. **Text Batching**: Adjacent cells with identical styling (color, bold, italic)
//!    are grouped into [`BatchedTextRun`]s for efficient text shaping.
//!
//! 3. **Default Background Skip**: Cells with the default background color don't
//!    generate separate background rectangles.
//!
//! 4. **Cell Measurement**: Font metrics are measured once using the '│' (BOX DRAWINGS
//!    LIGHT VERTICAL) character and cached for consistent cell dimensions.
//!
//! # Cell Dimensions
//!
//! Cell size is calculated from actual font metrics using the '│' character,
//! which spans the full cell height in properly designed terminal fonts:
//!
//! - **Width**: Measured from shaped '│' character
//! - **Height**: `(ascent + descent) × line_height_multiplier`
//!
//! The `line_height_multiplier` (default 1.0) can be adjusted to add extra
//! vertical space if needed for specific fonts.
//!
//! # Cursor Rendering
//!
//! The cursor shape comes from the grid (via [`GridSource::cursor`]) but can
//! be overridden per renderer through [`TerminalRenderer::cursor_shape_override`].
//! Supported shapes are `Block`, `Beam` (vertical bar), and `Underline`:
//!
//! - `Block` fills the entire cell.
//! - `Beam` and `Underline` use [`TerminalRenderer::cursor_thickness`] as a
//!   fraction of the cell's width or height respectively.
//!
//! Blinking is **not** handled here: the renderer receives an already-resolved
//! `cursor_visible` boolean from [`TerminalView`](crate::TerminalView), which
//! owns the blink timer.
//!
//! # Example
//!
//! ```ignore
//! use alacritty_terminal::vte::ansi::CursorShape;
//! use gpui::px;
//! use gpui_terminal::{ColorPalette, TerminalRenderer};
//!
//! let renderer = TerminalRenderer::new(
//!     "JetBrains Mono".to_string(),
//!     px(14.0),
//!     1.0,                                // line height multiplier
//!     0.15,                               // cursor thickness (15% of cell)
//!     ColorPalette::default(),
//!     Some(CursorShape::Beam),            // override shape, or `None`
//! );
//! ```

use crate::backend::{
    App, Bounds, Edges, Font, FontFeatures, FontStyle, FontWeight, Hsla, Pixels, Point,
    SharedString, Size, TextRun, UnderlineStyle, Window, paint_shaped_line, px, quad,
    transparent_black,
};
use crate::box_drawing;
use crate::colors::ColorPalette;
use crate::grid::GridSource;
use alacritty_terminal::vte::ansi::CursorShape;
use alacritty_terminal::{
    index::Line,
    term::{
        cell::{Cell, Flags},
        color::Colors,
    },
    vte::ansi::Color,
};

/// A batched run of text with consistent styling.
///
/// This struct groups adjacent terminal cells with identical visual attributes
/// to reduce the number of text rendering calls.
///
/// Runs are produced by [`TerminalRenderer::layout_row`] and consumed by the
/// paint loop. Adjacent cells extend an existing run when every styling field
/// matches; otherwise a new run is started, so a row can produce multiple runs
/// when styling changes (e.g. prompt text vs. output).
#[derive(Debug, Clone)]
pub struct BatchedTextRun {
    /// The text content to render
    pub text: String,

    /// Starting column position
    pub start_col: usize,

    /// Row position
    pub row: usize,

    /// Foreground color
    pub fg_color: Hsla,

    /// Background color
    pub bg_color: Hsla,

    /// Bold flag
    pub bold: bool,

    /// Italic flag
    pub italic: bool,

    /// Underline flag
    pub underline: bool,
}

/// Background rectangle to paint.
///
/// Represents a rectangular region with a solid color background.
///
/// Rects are produced by [`TerminalRenderer::layout_row`] and merged by
/// [`TerminalRenderer::merge_backgrounds`] before painting. The end column is
/// exclusive, so `end_col - start_col` is the width in cells.
#[derive(Debug, Clone)]
pub struct BackgroundRect {
    /// Starting column position
    pub start_col: usize,

    /// Ending column position (exclusive)
    pub end_col: usize,

    /// Row position
    pub row: usize,

    /// Background color
    pub color: Hsla,
}

impl BackgroundRect {
    /// Check if this rectangle can be merged with another.
    ///
    /// Two rectangles can be merged if they:
    /// - Are on the same row
    /// - Have the same color
    /// - Are horizontally adjacent (`self.end_col == other.start_col`)
    ///
    /// The merge is directional (self before other); callers iterating in
    /// left-to-right order can merge in place without checking the reverse.
    fn can_merge_with(&self, other: &Self) -> bool {
        self.row == other.row && self.color == other.color && self.end_col == other.start_col
    }
}

/// Terminal renderer with font settings and cell dimensions.
///
/// This struct manages the rendering of terminal content, including text,
/// backgrounds, and cursor. It maintains font metrics and provides the
/// [`paint`](Self::paint) method for drawing the terminal grid.
///
/// # Font Metrics
///
/// Cell dimensions are calculated from actual font measurements via
/// [`measure_cell`](Self::measure_cell). This ensures accurate character
/// positioning regardless of the font used.
///
/// # Cursor Style
///
/// The cursor shape is resolved as
/// `cursor_shape_override.unwrap_or(grid.cursor().shape)`, so the override
/// wins over what the program requests via `DECSCUSR`. `cursor_thickness`
/// only affects `Beam` and `Underline` (a `Block` fills the cell by
/// definition). Blinking is handled by the view, not by the renderer.
///
/// # Usage
///
/// The renderer is typically used internally by [`TerminalView`](crate::TerminalView),
/// but can also be used directly for custom rendering:
///
/// ```ignore
/// // Measure cell dimensions (call once per font change)
/// renderer.measure_cell(window);
///
/// // Paint the terminal grid
/// renderer.paint(bounds, padding, &term, None, true, false, window, cx);
/// ```
///
/// # Performance
///
/// For optimal performance:
/// - Call `measure_cell` only when font settings change
/// - The `paint` method is designed to be called every frame
/// - Background and text batching minimize GPU draw calls
#[derive(Clone)]
pub struct TerminalRenderer {
    /// Font family name (e.g., "Fira Code", "Menlo")
    pub font_family: String,

    /// Font size in pixels
    pub font_size: Pixels,

    /// Width of a single character cell
    pub cell_width: Pixels,

    /// Height of a single character cell (line height)
    pub cell_height: Pixels,

    /// Multiplier for line height to accommodate tall glyphs
    pub line_height_multiplier: f32,

    /// Color palette for resolving terminal colors
    pub palette: ColorPalette,

    /// Thickness of `Beam` and `Underline` cursors as a fraction of the
    /// cell's width or height respectively (e.g. `0.15` = 15%).
    ///
    /// Clamped to at least 1px so the cursor is always visible at small
    /// font sizes. Has no effect on `Block` cursors.
    pub cursor_thickness: f32,

    /// Forces the cursor shape, overriding the shape reported by the grid.
    ///
    /// - `None` → use the shape from `grid.cursor().shape` (i.e., whatever
    ///   the program requested via `DECSCUSR`).
    /// - `Some(shape)` → always draw this shape, ignoring the program.
    pub cursor_shape_override: Option<CursorShape>,
}

impl TerminalRenderer {
    /// Creates a new terminal renderer with the given font settings and color palette.
    ///
    /// Initial cell dimensions are approximated (`font_size * 0.6` wide,
    /// `font_size * 1.4` tall) and will be replaced by real measurements the
    /// first time [`measure_cell`](Self::measure_cell) runs, which happens at
    /// the start of the first render.
    ///
    /// # Arguments
    ///
    /// * `font_family` - The name of the font family to use
    /// * `font_size` - The font size in pixels
    /// * `line_height_multiplier` - Multiplier for line height (e.g., 1.2 for 20% extra)
    /// * `cursor_thickness` - Fraction of the cell occupied by `Beam`/`Underline`
    ///   cursors (e.g. `0.15` for 15%). Ignored for `Block`.
    /// * `palette` - The color palette to use for terminal colors
    /// * `cursor_shape_override` - If `Some`, forces this cursor shape and
    ///   ignores the grid's. If `None`, the grid's shape is used.
    ///
    /// # Returns
    ///
    /// A new `TerminalRenderer` instance with approximate cell dimensions.
    ///
    /// # Examples
    ///
    /// ```
    /// use alacritty_terminal::vte::ansi::CursorShape;
    /// use gpui::px;
    /// use gpui_terminal::render::TerminalRenderer;
    /// use gpui_terminal::ColorPalette;
    ///
    /// let renderer = TerminalRenderer::new(
    ///     "Fira Code".to_string(),
    ///     px(14.0),
    ///     1.0,                      // line height multiplier
    ///     0.15,                     // cursor thickness
    ///     ColorPalette::default(),
    ///     Some(CursorShape::Beam),  // or None to honor the program
    /// );
    /// ```
    pub fn new(
        font_family: String,
        font_size: Pixels,
        line_height_multiplier: f32,
        cursor_thickness: f32,
        palette: ColorPalette,
        cursor_shape_override: Option<CursorShape>,
    ) -> Self {
        // Default cell dimensions - will be measured on first paint
        // Using 0.6 as approximate em-width ratio for monospace fonts
        let cell_width = font_size * 0.6;
        let cell_height = font_size * 1.4; // Line height with some spacing

        Self {
            font_family,
            font_size,
            cell_width,
            cell_height,
            line_height_multiplier,
            cursor_thickness,
            cursor_shape_override,
            palette,
        }
    }

    /// Measure cell dimensions based on actual font metrics.
    ///
    /// This method measures the actual width and height of characters
    /// using the GPUI text system. It uses the '│' (BOX DRAWINGS LIGHT VERTICAL)
    /// character which spans the full cell height in properly designed terminal fonts.
    ///
    /// Only overrides the internal dimensions if the measured values are
    /// positive, so a font that fails to shape leaves the previous (or
    /// initial approximate) metrics in place rather than collapsing to zero.
    ///
    /// # Arguments
    ///
    /// * `window` - The GPUI window for text system access
    pub fn measure_cell(&mut self, window: &mut Window) {
        // Measure using '│' (U+2502, BOX DRAWINGS LIGHT VERTICAL)
        // This character spans the full cell height in terminal fonts, making it
        // ideal for measuring exact cell dimensions used by TUIs
        let font = Font {
            family: self.font_family.clone().into(),
            features: FontFeatures::default(),
            fallbacks: None,
            weight: FontWeight::NORMAL,
            style: FontStyle::Normal,
        };

        let text_run = TextRun {
            len: "│".len(),
            font,
            color: crate::backend::black(),
            background_color: None,
            underline: None,
            strikethrough: None,
        };

        // Shape the box-drawing character to get cell metrics
        let shaped = window
            .text_system()
            .shape_line("│".into(), self.font_size, &[text_run], None);

        // Get the width from the shaped line
        if shaped.width > px(0.0) {
            self.cell_width = shaped.width;
        }

        // Calculate height from ascent + descent with optional multiplier
        let line_height = (shaped.ascent + shaped.descent).ceil();
        if line_height > px(0.0) {
            self.cell_height = line_height * self.line_height_multiplier;
        }
    }

    /// Layout cells into batched text runs and background rects for a single row.
    ///
    /// This method processes a row of terminal cells and groups adjacent cells
    /// with identical styling into batched runs. It also collects background
    /// rectangles that need to be painted.
    ///
    /// # Behavior
    ///
    /// - **Wide-character spacers** (`Flags::WIDE_CHAR_SPACER`) are skipped, so
    ///   the preceding wide glyph occupies the pair of columns on its own.
    /// - **NUL cells** are normalized to spaces, so empty regions still get a
    ///   background rect and contribute to runs.
    /// - **Colors** are resolved through [`ColorPalette::resolve`] against the
    ///   grid's override colors (`colors`), which is where ANSI/program color
    ///   overrides take effect.
    /// - **Adjacent backgrounds** with the same color are extended in place,
    ///   and the final list is passed through
    ///   [`merge_backgrounds`](Self::merge_backgrounds) as a second pass.
    /// - **Adjacent text runs** are extended only when *all* styling fields
    ///   match (`fg_color`, `bg_color`, `bold`, `italic`, `underline`).
    ///
    /// # Arguments
    ///
    /// * `row` - The row number
    /// * `cells` - Iterator over `(column, Cell)` pairs
    /// * `colors` - Terminal color configuration
    ///
    /// # Returns
    ///
    /// A tuple of `(backgrounds, text_runs)` where:
    /// - `backgrounds` is a vector of merged background rectangles
    /// - `text_runs` is a vector of batched text runs
    pub fn layout_row(
        &self,
        row: usize,
        cells: impl Iterator<Item = (usize, Cell)>,
        colors: &Colors,
    ) -> (Vec<BackgroundRect>, Vec<BatchedTextRun>) {
        let mut backgrounds = Vec::new();
        let mut text_runs = Vec::new();

        let mut current_run: Option<BatchedTextRun> = None;
        let mut current_bg: Option<BackgroundRect> = None;

        for (col, cell) in cells {
            // Skip wide character spacers
            if cell.flags.contains(Flags::WIDE_CHAR_SPACER) {
                continue;
            }

            // Extract cell styling
            let fg_color = self.palette.resolve(cell.fg, colors);
            let bg_color = self.palette.resolve(cell.bg, colors);
            let bold = cell.flags.contains(Flags::BOLD);
            let italic = cell.flags.contains(Flags::ITALIC);
            let underline = cell.flags.contains(Flags::UNDERLINE);

            // Get the character (or space if empty)
            let ch = if cell.c == ' ' || cell.c == '\0' {
                ' '
            } else {
                cell.c
            };

            // Handle background rectangles
            if let Some(ref mut bg_rect) = current_bg {
                if bg_rect.color == bg_color && bg_rect.end_col == col {
                    // Extend current background
                    bg_rect.end_col = col + 1;
                } else {
                    // Save current background and start new one
                    backgrounds.push(bg_rect.clone());
                    current_bg = Some(BackgroundRect {
                        start_col: col,
                        end_col: col + 1,
                        row,
                        color: bg_color,
                    });
                }
            } else {
                // Start new background
                current_bg = Some(BackgroundRect {
                    start_col: col,
                    end_col: col + 1,
                    row,
                    color: bg_color,
                });
            }

            // Handle text runs
            if let Some(ref mut run) = current_run {
                if run.fg_color == fg_color
                    && run.bg_color == bg_color
                    && run.bold == bold
                    && run.italic == italic
                    && run.underline == underline
                {
                    // Extend current run
                    run.text.push(ch);
                } else {
                    // Save current run and start new one
                    text_runs.push(run.clone());
                    current_run = Some(BatchedTextRun {
                        text: ch.to_string(),
                        start_col: col,
                        row,
                        fg_color,
                        bg_color,
                        bold,
                        italic,
                        underline,
                    });
                }
            } else {
                // Start new run
                current_run = Some(BatchedTextRun {
                    text: ch.to_string(),
                    start_col: col,
                    row,
                    fg_color,
                    bg_color,
                    bold,
                    italic,
                    underline,
                });
            }
        }

        // Push final run and background
        if let Some(run) = current_run {
            text_runs.push(run);
        }
        if let Some(bg) = current_bg {
            backgrounds.push(bg);
        }

        // Merge adjacent backgrounds with same color
        let merged_backgrounds = self.merge_backgrounds(backgrounds);

        (merged_backgrounds, text_runs)
    }

    /// Merge adjacent background rects with same color.
    ///
    /// This optimization reduces the number of rectangles to paint by
    /// combining horizontally adjacent rectangles that share the same color.
    ///
    /// Runs in O(n) over an already-sorted-by-column list, using
    /// [`BackgroundRect::can_merge_with`] to decide adjacency. The first rect
    /// is consumed with `remove(0)`, so callers can pass a vector they no
    /// longer need (as [`layout_row`](Self::layout_row) does).
    ///
    /// # Arguments
    ///
    /// * `rects` - Vector of background rectangles to merge, in paint order
    ///
    /// # Returns
    ///
    /// A new vector with merged rectangles
    fn merge_backgrounds(&self, mut rects: Vec<BackgroundRect>) -> Vec<BackgroundRect> {
        if rects.is_empty() {
            return rects;
        }

        let mut merged = Vec::new();
        let mut current = rects.remove(0);

        for rect in rects {
            if current.can_merge_with(&rect) {
                current.end_col = rect.end_col;
            } else {
                merged.push(current);
                current = rect;
            }
        }

        merged.push(current);
        merged
    }

    /// Paint terminal content to the window.
    ///
    /// This is the main rendering method that draws the terminal grid,
    /// including backgrounds, text, and cursor.
    ///
    /// # Paint order
    ///
    /// 1. Default background over the full `bounds` (covers the padding area).
    /// 2. Selection highlight, per visible line, before any text.
    /// 3. Non-default background rectangles (batched per row).
    /// 4. Box-drawing characters, split into two passes:
    ///    - horizontal spans across adjacent cells (avoids 1px gaps between
    ///      cells that would otherwise appear at arbitrary font sizes),
    ///    - vertical components (or the full glyph, if not part of a span).
    /// 5. Regular text characters, one shape call per cell.
    /// 6. Cursor, if the grid reports one and it should be visible this frame.
    ///
    /// # Cursor visibility
    ///
    /// `cursor_visible` is the current phase of the blink cycle (owned by the
    /// caller). `cursor_blink` is the **effective** blink state after applying
    /// [`TerminalConfig::cursor_blink_override`](crate::TerminalConfig). The
    /// cursor is painted when `cursor_visible || !cursor_blink`:
    ///
    /// - Non-blinking cursor (`cursor_blink == false`) → always painted.
    /// - Blinking cursor in the "on" phase (`cursor_visible == true`) → painted.
    /// - Blinking cursor in the "off" phase → skipped.
    ///
    /// The shape is resolved as `self.cursor_shape_override.unwrap_or(cursor.shape)`,
    /// and `Beam`/`Underline` use [`cursor_thickness`](Self::cursor_thickness).
    /// Variants other than `Block`/`Beam`/`Underline` (e.g. `Hidden`) cause the
    /// cursor to be skipped.
    ///
    /// # Selection
    ///
    /// `selection` is a pair of grid points in buffer coordinates. They are
    /// normalized (swapped if needed) and clamped per line so that a multi-line
    /// selection fills the middle lines edge to edge. The highlight is painted
    /// only for the visible rows that intersect the range.
    ///
    /// # Arguments
    ///
    /// * `bounds` - The bounding box to render within
    /// * `padding` - Padding around the terminal content
    /// * `grid` - The cells to draw: a [`Term`](alacritty_terminal::term::Term),
    ///   or any other [`GridSource`]
    /// * `selection` - Normalized-on-the-fly `(start, end)` grid points to
    ///   highlight, or `None` for no selection
    /// * `cursor_visible` - Current blink phase (`true` = cursor on this frame)
    /// * `cursor_blink` - Effective blink state after overrides; controls
    ///   whether `cursor_visible == false` actually hides the cursor
    /// * `window` - The GPUI window
    /// * `cx` - The application context
    #[allow(clippy::too_many_arguments)]
    pub fn paint(
        &self,
        bounds: Bounds<Pixels>,
        padding: Edges<Pixels>,
        grid: &impl GridSource,
        selection: Option<(
            alacritty_terminal::index::Point,
            alacritty_terminal::index::Point,
        )>,
        cursor_visible: bool,
        cursor_blink: bool,
        window: &mut Window,
        _cx: &mut App,
    ) {
        // Get terminal dimensions
        let num_lines = grid.screen_lines();
        let num_cols = grid.columns();
        let colors = grid.colors();
        let display_offset = grid.display_offset() as i32;

        // Calculate default background color
        let default_bg = self.palette.resolve(
            Color::Named(alacritty_terminal::vte::ansi::NamedColor::Background),
            colors,
        );

        // Paint default background (covers full bounds including padding)
        window.paint_quad(quad(
            bounds,
            px(0.0),
            default_bg,
            Edges::<Pixels>::default(),
            transparent_black(),
            Default::default(),
        ));

        // Calculate origin offset (content starts after padding)
        let origin = Point {
            x: bounds.origin.x + padding.left,
            y: bounds.origin.y + padding.top,
        };

        let sel = selection.map(|(mut start, mut end)| {
            if start > end {
                std::mem::swap(&mut start, &mut end);
            }
            (start, end)
        });

        // Iterate over visible lines
        for line_idx in 0..num_lines {
            let line = Line(line_idx as i32 - display_offset);

            if let Some((start, end)) = sel
                && line >= start.line
                && line <= end.line
            {
                let start_col = if line == start.line {
                    start.column.0
                } else {
                    0
                };
                let end_col = if line == end.line {
                    end.column.0
                } else {
                    num_cols - 1
                };

                if start_col <= end_col {
                    let x = origin.x + self.cell_width * (start_col as f32);
                    let y = origin.y + self.cell_height * (line_idx as f32);
                    let width = self.cell_width * ((end_col - start_col + 1) as f32);
                    let height = self.cell_height;

                    let rect_bounds = Bounds {
                        origin: Point { x, y },
                        size: Size { width, height },
                    };

                    window.paint_quad(quad(
                        rect_bounds,
                        px(0.0),
                        self.palette.selection(),
                        Edges::<Pixels>::default(),
                        transparent_black(),
                        Default::default(),
                    ));
                }
            }

            // Collect cells for this line
            let cells: Vec<(usize, Cell)> = (0..num_cols)
                .map(|col_idx| (col_idx, grid.cell(line_idx, col_idx)))
                .collect();

            // Layout the row for backgrounds
            let (backgrounds, _) = self.layout_row(line_idx, cells.iter().cloned(), colors);

            // Paint backgrounds
            for bg_rect in backgrounds {
                // Skip if it's the default background color
                if bg_rect.color == default_bg {
                    continue;
                }

                let x = origin.x + self.cell_width * (bg_rect.start_col as f32);
                let y = origin.y + self.cell_height * (bg_rect.row as f32);
                let width = self.cell_width * ((bg_rect.end_col - bg_rect.start_col) as f32);
                let height = self.cell_height;

                let rect_bounds = Bounds {
                    origin: Point { x, y },
                    size: Size { width, height },
                };

                window.paint_quad(quad(
                    rect_bounds,
                    px(0.0),
                    bg_rect.color,
                    Edges::<Pixels>::default(),
                    transparent_black(),
                    Default::default(),
                ));
            }

            // Calculate vertical offset to center text in cell
            // The multiplier adds extra height; we want to distribute it evenly top/bottom
            let base_height = self.cell_height / self.line_height_multiplier;
            let vertical_offset = (self.cell_height - base_height) / 2.0;

            let y_base = origin.y + self.cell_height * (line_idx as f32);
            let cy = y_base + self.cell_height / 2.0;

            // Use cells vec for multiple passes (already collected above)
            let cells_vec = &cells;

            // First pass: find and draw horizontal spans of box-drawing characters
            // This draws continuous lines across multiple cells to avoid gaps
            let mut processed_horizontal: std::collections::HashSet<usize> =
                std::collections::HashSet::new();

            let mut i = 0;
            while i < cells_vec.len() {
                let (col_idx, ref cell) = cells_vec[i];
                let ch = cell.c;

                // Check if this starts a horizontal span
                if let Some(weight) = box_drawing::get_horizontal_weight(ch) {
                    let fg_color = self.palette.resolve(cell.fg, colors);
                    let start_col = col_idx;
                    let mut end_col = col_idx;

                    // Look ahead for consecutive cells with same horizontal weight
                    let mut j = i + 1;
                    while j < cells_vec.len() {
                        let (next_col, ref next_cell) = cells_vec[j];
                        // Must be adjacent
                        if next_col != end_col + 1 {
                            break;
                        }
                        // Must have same horizontal weight and same color
                        let next_fg = self.palette.resolve(next_cell.fg, colors);
                        if box_drawing::get_horizontal_weight(next_cell.c) == Some(weight)
                            && next_fg == fg_color
                        {
                            end_col = next_col;
                            j += 1;
                        } else {
                            break;
                        }
                    }

                    // Draw the horizontal span
                    let start_x = origin.x + self.cell_width * (start_col as f32);
                    let end_x = origin.x + self.cell_width * ((end_col + 1) as f32);

                    box_drawing::draw_horizontal_span(
                        start_x,
                        end_x,
                        cy,
                        weight,
                        self.cell_width,
                        fg_color,
                        window,
                    );

                    // Mark these columns as having horizontal drawn
                    for col in start_col..=end_col {
                        processed_horizontal.insert(col);
                    }

                    // Skip past this span
                    i = j;
                    continue;
                }
                i += 1;
            }

            // Second pass: draw vertical components and non-horizontal box chars
            for (col_idx, cell) in cells_vec.iter() {
                let ch = cell.c;

                if ch == ' ' || ch == '\0' {
                    continue;
                }

                let x = origin.x + self.cell_width * (*col_idx as f32);
                let fg_color = self.palette.resolve(cell.fg, colors);

                if box_drawing::is_box_drawing_char(ch) {
                    let cell_bounds = Bounds {
                        origin: Point { x, y: y_base },
                        size: Size {
                            width: self.cell_width,
                            height: self.cell_height,
                        },
                    };

                    if processed_horizontal.contains(col_idx) {
                        // Horizontal already drawn, just draw vertical components
                        box_drawing::draw_vertical_components(
                            ch,
                            cell_bounds,
                            fg_color,
                            self.cell_width,
                            window,
                        );
                    } else {
                        // Not part of a horizontal span, draw the whole character
                        box_drawing::draw_box_character(
                            ch,
                            cell_bounds,
                            fg_color,
                            self.cell_width,
                            window,
                        );
                    }
                    continue;
                }
            }

            // Third pass: draw regular text characters
            for (col_idx, cell) in cells_vec.iter() {
                let ch = cell.c;

                // Skip empty cells and box-drawing (already handled)
                if ch == ' ' || ch == '\0' || box_drawing::is_box_drawing_char(ch) {
                    continue;
                }

                let x = origin.x + self.cell_width * (*col_idx as f32);
                let fg_color = self.palette.resolve(cell.fg, colors);

                // For regular text, apply vertical offset for centering
                let y = y_base + vertical_offset;

                // Get cell flags for styling
                let flags = cell.flags;
                let bold = flags.contains(alacritty_terminal::term::cell::Flags::BOLD);
                let italic = flags.contains(alacritty_terminal::term::cell::Flags::ITALIC);
                let underline = flags.contains(alacritty_terminal::term::cell::Flags::UNDERLINE);

                // Create font with styling
                let font = Font {
                    family: self.font_family.clone().into(),
                    features: FontFeatures::default(),
                    fallbacks: None,
                    weight: if bold {
                        FontWeight::BOLD
                    } else {
                        FontWeight::NORMAL
                    },
                    style: if italic {
                        FontStyle::Italic
                    } else {
                        FontStyle::Normal
                    },
                };

                // Create text run for this single character
                let char_str = ch.to_string();
                let text_run = TextRun {
                    len: char_str.len(),
                    font,
                    color: fg_color,
                    background_color: None,
                    underline: if underline {
                        Some(UnderlineStyle {
                            thickness: px(1.0),
                            color: Some(fg_color),
                            wavy: false,
                        })
                    } else {
                        None
                    },
                    strikethrough: None,
                };

                // Shape and paint the character
                let text: SharedString = char_str.into();
                let shaped_line =
                    window
                        .text_system()
                        .shape_line(text, self.font_size, &[text_run], None);

                // Paint at exact cell position (ignore errors)
                let _ =
                    paint_shaped_line(&shaped_line, Point { x, y }, self.cell_height, window, _cx);
            }
        }

        // Paint cursor
        if let Some(cursor) = grid.cursor() {
            let should_paint = cursor_visible || !cursor_blink;

            if !should_paint {
                return;
            }

            let x = origin.x + self.cell_width * (cursor.column as f32);
            let y = origin.y + self.cell_height * (cursor.line as f32);
            let thickness = self.cursor_thickness;

            let shape = self.cursor_shape_override.unwrap_or(cursor.shape);

            let cursor_bounds = match shape {
                CursorShape::Block => Bounds {
                    origin: Point { x, y },
                    size: Size {
                        width: self.cell_width,
                        height: self.cell_height,
                    },
                },
                CursorShape::Beam => {
                    let w = (self.cell_width * thickness).max(px(1.0));
                    Bounds {
                        origin: Point { x, y },
                        size: Size {
                            width: w,
                            height: self.cell_height,
                        },
                    }
                }
                CursorShape::Underline => {
                    let h = (self.cell_height * thickness).max(px(1.0));
                    Bounds {
                        origin: Point {
                            x,
                            y: y + self.cell_height - h,
                        },
                        size: Size {
                            width: self.cell_width,
                            height: h,
                        },
                    }
                }
                _ => return, // Hidden or variants
            };

            let cursor_color = self.palette.resolve(
                Color::Named(alacritty_terminal::vte::ansi::NamedColor::Cursor),
                colors,
            );

            window.paint_quad(quad(
                cursor_bounds,
                px(0.0),
                cursor_color,
                Edges::<Pixels>::default(),
                transparent_black(),
                Default::default(),
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_renderer_creation() {
        let renderer = TerminalRenderer::new(
            "Fira Code".to_string(),
            px(14.0),
            1.0,
            0.15,
            ColorPalette::default(),
            None,
        );
        assert_eq!(renderer.font_family, "Fira Code");
        assert_eq!(renderer.font_size, px(14.0));
        assert_eq!(renderer.line_height_multiplier, 1.0);
    }

    #[test]
    fn test_background_rect_merge() {
        let black = Hsla::black();

        let rect1 = BackgroundRect {
            start_col: 0,
            end_col: 5,
            row: 0,
            color: black,
        };

        let rect2 = BackgroundRect {
            start_col: 5,
            end_col: 10,
            row: 0,
            color: black,
        };

        assert!(rect1.can_merge_with(&rect2));

        let rect3 = BackgroundRect {
            start_col: 5,
            end_col: 10,
            row: 1,
            color: black,
        };

        assert!(!rect1.can_merge_with(&rect3));
    }

    #[test]
    fn test_merge_backgrounds() {
        let renderer = TerminalRenderer::new(
            "monospace".to_string(),
            px(14.0),
            1.0,
            0.15,
            ColorPalette::default(),
            None,
        );
        let black = Hsla::black();

        let rects = vec![
            BackgroundRect {
                start_col: 0,
                end_col: 5,
                row: 0,
                color: black,
            },
            BackgroundRect {
                start_col: 5,
                end_col: 10,
                row: 0,
                color: black,
            },
        ];

        let merged = renderer.merge_backgrounds(rects);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].start_col, 0);
        assert_eq!(merged[0].end_col, 10);
    }
}
