//! What the renderer draws: a grid of cells.
//!
//! [`GridSource`] is implemented for alacritty's [`Term`], which is what
//! [`TerminalView`](crate::TerminalView) draws. It lets the renderer draw a
//! grid that lives somewhere else too, such as a terminal emulated in another
//! process or on another machine whose cells arrive over a connection.
//! [`CellGrid`] holds such a grid.
//!
//! # Coordinate model
//!
//! Two coordinate systems meet here:
//!
//! - **Visible rows**: `0..screen_lines()`, where `0` is the top row currently
//!   on screen. This is what [`GridSource::cell`] accepts.
//! - **Buffer rows**: alacritty's `Line(i32)`, where `0` is the top of the
//!   live screen buffer and negative values reach into the scrollback.
//!
//! [`GridSource::display_offset`] is the bridge: a backend that has scrollback
//! must subtract it when translating a visible row into a buffer row. The
//! renderer does not do this subtraction itself — the contract is documented on
//! the method.

use alacritty_terminal::{
    grid::Dimensions,
    index::{Column, Line, Point},
    term::{Term, TermMode, cell::Cell, color::Colors},
    vte::ansi::CursorShape,
};

/// Description of the cursor that the renderer should draw.
///
/// This is the cursor's *appearance and position*, not its visibility phase:
/// blinking is driven by [`TerminalView`](crate::TerminalView), which decides
/// per frame whether to paint it. The renderer just needs to know where and
/// how to draw it when asked.
///
/// The shape and blink flag come from the program via `DECSCUSR`; the host
/// application can override them through
/// [`TerminalConfig::cursor_shape_override`](crate::TerminalConfig::cursor_shape_override)
/// and
/// [`TerminalConfig::cursor_blink_override`](crate::TerminalConfig::cursor_blink_override).
#[derive(Debug, Clone, Copy)]
pub struct CursorInfo {
    /// Visible row of the cursor (`0` = top row on screen).
    ///
    /// Already translated to visible coordinates; the renderer positions the
    /// cursor with `cell_height * line` directly.
    pub line: usize,

    /// Column of the cursor (0-indexed, left to right).
    pub column: usize,

    /// Cursor shape requested by the program.
    ///
    /// Common values are `Block`, `Beam`, and `Underline`; the renderer may
    /// replace this with an override from the config. Other variants (e.g.
    /// `Hidden`) are treated as "draw nothing" by the renderer.
    pub shape: CursorShape,

    /// Whether the program requested the cursor to blink.
    ///
    /// The view combines this with
    /// [`TerminalConfig::cursor_blink_override`](crate::TerminalConfig::cursor_blink_override)
    /// to obtain the *effective* blink state, and that effective state (not
    /// this field) is what gates the renderer's paint call. If you read this
    /// field directly inside `paint`, blinking will break when an override is
    /// set.
    pub blink: bool,
}

/// A grid of cells the renderer can draw.
///
/// Implementors provide the visible rows of a terminal grid, the colors set by
/// the program, and the current cursor. The renderer only reads from this
/// trait; it never mutates the grid.
///
/// # Implementations
///
/// - [`Term`] from `alacritty_terminal` — the normal case, backed by a live
///   VTE parser and a real scrollback buffer.
/// - [`CellGrid`] — an in-memory grid you can populate cell by cell, useful
///   for cells that arrive from another process or machine.
///
/// # Contract for `cell`
///
/// `cell(line, column)` receives **visible** rows: `0` is the top row
/// currently on screen. If the backend has scrollback and the user scrolled
/// back, the implementation must subtract [`display_offset`](Self::display_offset)
/// internally so that `cell(0, _)` refers to what the user actually sees at
/// the top. Implementations without scrollback (like [`CellGrid`]) always
/// report `display_offset() == 0`, which makes the subtraction a no-op.
///
/// # Adding a new backend
///
/// To draw, say, a grid streamed over the network, implement this trait on a
/// type that stores the cells. `TerminalRenderer::paint` accepts `&impl
/// GridSource`, so the renderer does not need any changes:
///
/// ```ignore
/// renderer.paint(bounds, padding, &my_grid, None, true, false, window, cx);
/// ```
pub trait GridSource {
    /// Number of rows on screen.
    ///
    /// This is the *visible* height of the grid, not the total buffer size.
    /// Scrollback lives behind [`display_offset`](Self::display_offset), not
    /// in this count.
    fn screen_lines(&self) -> usize;

    /// Number of columns.
    fn columns(&self) -> usize;

    /// The cell at a visible row and column.
    ///
    /// `line` is a visible row (`0` = top of what the user sees right now);
    /// the implementation is responsible for translating it into buffer
    /// coordinates using [`display_offset`](Self::display_offset). `column`
    /// is a 0-indexed column.
    ///
    /// The cell is returned by value, so implementations are free to hold
    /// their storage behind whatever representation they like.
    fn cell(&self, line: usize, column: usize) -> Cell;

    /// Colors set by the program, which override the palette.
    ///
    /// These are colors written by the running program (via OSC 4, OSC 10/11,
    /// or the 256-color and truecolor escape sequences). The renderer resolves
    /// each cell's color against this table first, falling back to
    /// [`TerminalRenderer::palette`](crate::render::TerminalRenderer) when the
    /// program has not set one.
    fn colors(&self) -> &Colors;

    /// Where and how to draw the cursor, or `None` while it is hidden.
    ///
    /// Returns `None` when the program hid the cursor or the terminal has no
    /// cursor. When `Some`, the position is already in visible-row
    /// coordinates and the shape/blink reflect what the program requested
    /// (`DECSCUSR`). Overrides from
    /// [`TerminalConfig`](crate::TerminalConfig) are applied by the caller,
    /// not here.
    fn cursor(&self) -> Option<CursorInfo>;

    /// How many lines the view is scrolled up from the bottom of the buffer.
    ///
    /// This is the distance, in rows, between the bottommost visible row and
    /// the last line written by the program. It follows alacritty's convention:
    ///
    /// - `0` → the view is pinned to the bottom (the "live" end); whatever the
    ///   program writes shows up immediately.
    /// - `N > 0` → the user has scrolled `N` rows back into the scrollback.
    ///
    /// The renderer uses this value for two things:
    ///
    /// 1. **Selection**: when translating on-screen positions into buffer
    ///    coordinates (`Line(line_idx as i32 - display_offset)`).
    /// 2. **Cursor**: it is only painted when this is `0`, because a cursor
    ///    shown over the scrollback would be misleading (the process is not
    ///    writing there).
    ///
    /// # Contract with [`cell`](Self::cell)
    ///
    /// `cell(line, column)` receives **visible** rows (`0` = top of what the
    /// user sees right now). The implementation is responsible for subtracting
    /// this offset to reach the internal buffer coordinate. The renderer does
    /// not do it on its own.
    fn display_offset(&self) -> usize;
}

impl<T> GridSource for Term<T> {
    fn screen_lines(&self) -> usize {
        self.grid().screen_lines()
    }

    fn columns(&self) -> usize {
        self.grid().columns()
    }

    fn cell(&self, line: usize, column: usize) -> Cell {
        let offset = self.grid().display_offset() as i32;
        let display_line = Line(line as i32 - offset);
        self.grid()[Point::new(display_line, Column(column))].clone()
    }

    fn colors(&self) -> &Colors {
        Term::colors(self)
    }

    fn cursor(&self) -> Option<CursorInfo> {
        if !self.mode().contains(TermMode::SHOW_CURSOR) {
            return None;
        }
        let point = self.grid().cursor.point;
        let cursor_style = self.cursor_style();
        Some(CursorInfo {
            line: point.line.0.max(0) as usize,
            column: point.column.0,
            shape: cursor_style.shape,
            blink: cursor_style.blinking,
        })
    }

    fn display_offset(&self) -> usize {
        self.grid().display_offset()
    }
}

/// A grid of cells kept outside any emulator, for drawing cells that come
/// from elsewhere.
///
/// Use this when you have cell data that did not come from a local VTE parser
/// — for example, a terminal mirrored from another process or machine, a
/// synthetic grid for tests, or a static preview. Populate it via
/// [`set_line`](Self::set_line) or [`cell_mut`](Self::cell_mut), then hand it
/// to [`TerminalRenderer::paint`](crate::render::TerminalRenderer::paint)
/// exactly like a [`Term`].
///
/// # No scrollback
///
/// `CellGrid` stores only the visible rows: `cells.len() == lines * columns`.
/// There is nothing behind row `0` to scroll into, so
/// [`display_offset`](GridSource::display_offset) always returns `0` and
/// [`cell`](GridSource::cell) does not subtract anything. If you need
/// scrollback, apply it before calling [`set_line`](Self::set_line): the grid
/// you build should always represent what the user is currently looking at.
///
/// # Example
///
/// ```ignore
/// use alacritty_terminal::term::cell::Cell;
/// use gpui_terminal::{CellGrid, GridSource};
///
/// let mut grid = CellGrid::new(24, 80);
/// grid.set_line(0, "hello".chars().map(|c| Cell { c, ..Cell::default() }));
/// grid.set_cursor(Some(CursorInfo {
///     line: 0,
///     column: 5,
///     shape: CursorShape::Beam,
///     blink: true,
/// }));
/// ```
#[derive(Clone)]
pub struct CellGrid {
    /// Number of rows.
    lines: usize,
    /// Number of columns.
    columns: usize,
    /// Row-major cells: `cells[line * columns + column]`.
    cells: Vec<Cell>,
    /// Program-set colors, forwarded to the renderer as-is.
    colors: Colors,
    /// Cursor, or `None` for hidden.
    cursor: Option<CursorInfo>,
}

impl CellGrid {
    /// A blank grid of `lines` rows and `columns` columns.
    ///
    /// Every cell starts as [`Cell::default()`] (a space with default styling)
    /// and the cursor is hidden. Fill in the content with
    /// [`set_line`](Self::set_line) or [`cell_mut`](Self::cell_mut) before
    /// drawing.
    pub fn new(lines: usize, columns: usize) -> Self {
        Self {
            lines,
            columns,
            cells: vec![Cell::default(); lines * columns],
            colors: Colors::default(),
            cursor: None,
        }
    }

    /// Changes the size, keeping the cells that still fit.
    ///
    /// Cells outside the overlap between the old and new rectangle are
    /// dropped; cells inside it are preserved. New cells are
    /// [`Cell::default()`]. This matches "top-left anchored" resize, which is
    /// what you usually want when the host window grows or shrinks.
    pub fn resize(&mut self, lines: usize, columns: usize) {
        let mut cells = vec![Cell::default(); lines * columns];
        for line in 0..lines.min(self.lines) {
            for column in 0..columns.min(self.columns) {
                cells[line * columns + column] = self.cells[line * self.columns + column].clone();
            }
        }
        self.lines = lines;
        self.columns = columns;
        self.cells = cells;
    }

    /// The cell at a row and column, to change it.
    ///
    /// Unlike [`GridSource::cell`], this takes the raw storage index: no
    /// display-offset translation, because `CellGrid` has no scrollback. Panics
    /// if `line >= lines` or `column >= columns`.
    pub fn cell_mut(&mut self, line: usize, column: usize) -> &mut Cell {
        &mut self.cells[line * self.columns + column]
    }

    /// Replaces a whole row; cells past the end of `cells` become blank.
    ///
    /// Passing fewer cells than `columns` zeroes the remainder of the row
    /// (via [`Cell::default`]); passing more is truncated. Equivalent to
    /// writing each cell in order, but cheaper for full-row updates.
    pub fn set_line(&mut self, line: usize, cells: impl IntoIterator<Item = Cell>) {
        let row = &mut self.cells[line * self.columns..(line + 1) * self.columns];
        let mut given = cells.into_iter();
        for slot in row {
            *slot = given.next().unwrap_or_default();
        }
    }

    /// Colors the program set, which override the palette.
    ///
    /// Mutating this is how you make the grid report program-set colors to the
    /// renderer. The renderer resolves each cell's color against this table
    /// first, falling back to its own palette when the program has not set a
    /// color.
    pub fn colors_mut(&mut self) -> &mut Colors {
        &mut self.colors
    }

    /// Sets the cursor, or hides it with `None`.
    ///
    /// The [`CursorInfo`] is stored as-is and returned by
    /// [`GridSource::cursor`]. Position is in visible-row coordinates, so pass
    /// rows that already reflect any scrolling you applied when filling the
    /// grid.
    pub fn set_cursor(&mut self, cursor: Option<CursorInfo>) {
        self.cursor = cursor;
    }
}

impl GridSource for CellGrid {
    /// Returns the number of rows this grid was constructed with.
    fn screen_lines(&self) -> usize {
        self.lines
    }

    /// Returns the number of columns this grid was constructed with.
    fn columns(&self) -> usize {
        self.columns
    }

    /// Returns the cell at a visible row and column.
    ///
    /// No scrollback means no offset to subtract: `line` is used directly as
    /// the storage row. The caller is expected to have already applied any
    /// scrolling when building the grid (see [`CellGrid`]'s docs).
    fn cell(&self, line: usize, column: usize) -> Cell {
        self.cells[line * self.columns + column].clone()
    }

    /// Returns the program-set colors stored on the grid.
    fn colors(&self) -> &Colors {
        &self.colors
    }

    /// Returns the stored cursor, or `None` if it was hidden via
    /// [`set_cursor`](CellGrid::set_cursor).
    fn cursor(&self) -> Option<CursorInfo> {
        self.cursor
    }

    /// Always `0`: [`CellGrid`] does not keep scrollback, only the visible
    /// rows. Whoever feeds it (e.g. from another machine or process) must
    /// apply scrolling before calling [`set_line`](CellGrid::set_line), so
    /// that `cell(0, 0)` is always the first row on screen.
    ///
    /// Supporting scroll here would require storing scrollback beyond
    /// `cells`; until then there is nothing for [`cell`](Self::cell) to index
    /// into when subtracting a nonzero offset.
    fn display_offset(&self) -> usize {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_cell_grid_keeps_what_fits_when_resized() {
        let mut grid = CellGrid::new(2, 3);
        grid.set_line(
            0,
            "abc".chars().map(|c| Cell {
                c,
                ..Cell::default()
            }),
        );
        grid.resize(3, 2);
        assert_eq!(grid.cell(0, 0).c, 'a');
        assert_eq!(grid.cell(0, 1).c, 'b');
        assert_eq!(grid.cell(2, 1).c, ' ');
    }
}
