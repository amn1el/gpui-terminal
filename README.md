# gpui-terminal
![Crates.io Version](https://img.shields.io/crates/v/gpui-terminal?logo=rust)

A terminal emulator component for [GPUI](https://gpui.rs) applications. Uses [alacritty_terminal](https://docs.rs/alacritty_terminal) for VTE parsing.

<img width="1763" alt="Image" src="https://github.com/user-attachments/assets/713ed2b3-e08d-4ff5-8aeb-2d02660ecf7d" />

## Features

- Full ANSI escape sequence support
- 16 ANSI colors, 256-color mode, and 24-bit true color
- Bold, italic, and underline text styles
- Keyboard input with application cursor mode (vim/tmux compatible)
- Clipboard integration via OSC 52
- Dynamic configuration (font size, colors)
- Push-based async I/O
- Accepts any `Read`/`Write` streams (not tied to a specific PTY library)
- Mouse text selection with copy to clipboard
- Pluggable grid source: draw `alacritty_terminal::Term` or any custom `GridSource`

## Usage

```rust
use gpui_terminal::{TerminalConfig, TerminalView, ColorPalette};

// Create terminal with PTY reader/writer
let terminal = cx.new(|cx| {
    TerminalView::new(pty_writer, pty_reader, config, cx)
        .with_resize_callback(|cols, rows| {
            // Notify PTY of size change
        })
        .with_exit_callback(|_window, cx| {
            cx.quit();
        })
});
```

## Custom grids

The renderer draws anything that implements `GridSource`, not just
`alacritty_terminal::Term`. This makes it possible to render cells that
come from another process or machine without spinning up a local VTE
parser.

`CellGrid` is a ready-made in-memory `GridSource` for that case:

```rust
use gpui_terminal::{CellGrid, GridSource};
use alacritty_terminal::term::cell::Cell;

let mut grid = CellGrid::new(24, 80);

// Replace a whole row at once; cells past the end become blank.
grid.set_line(0, "hello from elsewhere".chars().map(|c| Cell {
    c,
    ..Cell::default()
}));

grid.set_cursor(Some((0, 20)));

// Hand it to the renderer exactly like a `Term`.
renderer.paint(bounds, padding, &grid, None, window, cx);
```
`display_offset` semantics
Both `CellGrid` and any custom implementation must respect the contract
defined on `GridSource::display_offset`:

- `0` means the view is pinned to the bottom of the buffer. `cell(0, _)`
is the top visible row.

- `N > 0` means the user has scrolled `N` rows back into the scrollback.

`cell(line, column)` always receives visible rows; the implementation
is responsible for subtracting `display_offset` before indexing its
internal buffer. `Term` does this internally, so scrollback works out of
the box. `CellGrid` has no scrollback and always reports `0`; whoever
feeds it must apply scrolling before calling `set_line`.

See `src/main.rs` for a complete example using `portable-pty`, and
`src/grid.rs` for the `GridSource` trait and `CellGrid`.

## Configuration

```rust
let colors = ColorPalette::builder()
    .background(0x16, 0x16, 0x17)
    .foreground(0xC9, 0xC7, 0xCD)
    .cursor(0xC9, 0xC7, 0xCD)
    .black(0x10, 0x10, 0x10)
    // ... other colors
    .build();

let config = TerminalConfig {
    font_family: "Mononoki Nerd Font".into(),
    font_size: px(14.0),
    cols: 80,
    rows: 24,
    scrollback: 10000,
    line_height_multiplier: 1.0,
    padding: Edges::all(px(8.0)),
    copy_on_selection: true,
    colors,
};
```

## Current Limitations

- No built-in keybindings or scrollbar for scrollback. The renderer honors
  `display_offset`, so scrolling works if the host application sets it on
  the `GridSource`, but `TerminalView` does not wire up mouse wheel or
  `PageUp`/`PageDown` on its own.

## License

MIT OR Apache-2.0
