#[cfg(feature = "gpui")]
mod inner {
    pub use gpui::*;

    pub fn paint_shaped_line(
        line: &ShapedLine,
        origin: Point<Pixels>,
        line_height: Pixels,
        window: &mut Window,
        cx: &mut App,
    ) -> Result<()> {
        line.paint(origin, line_height, window, cx)
    }

    pub fn attach_scroll<E, V>(
        el: E,
        handler: impl Fn(&mut V, f32, &mut Window, &mut Context<V>) + 'static,
        cx: &mut Context<V>,
    ) -> E
    where
        E: InteractiveElement,
        V: 'static,
    {
        el.on_scroll_wheel(
            cx.listener(move |this, event: &ScrollWheelEvent, window, cx| {
                let delta_y = match event.delta {
                    ScrollDelta::Lines(point) => point.y,
                    ScrollDelta::Pixels(point) => f32::from(point.y) / 20.0,
                };
                handler(this, delta_y, window, cx);
            }),
        )
    }
}

#[cfg(feature = "gpui-kit")]
mod inner {
    pub use gpui_kit::*;

    pub fn paint_shaped_line(
        line: &ShapedLine,
        origin: Point<Pixels>,
        line_height: Pixels,
        window: &mut Window,
        cx: &mut App,
    ) -> Result<()> {
        line.paint(origin, line_height, TextAlign::Left, None, window, cx)
    }

    pub fn attach_scroll<E, V>(
        el: E,
        handler: impl Fn(&mut V, f32, &mut Window, &mut Context<V>) + 'static,
        cx: &mut Context<V>,
    ) -> E
    where
        E: InteractiveElement,
        V: 'static,
    {
        el.on_scroll(cx.listener(move |this, event: &ScrollEvent, window, cx| {
            let delta_y = match event.delta {
                ScrollDelta::Lines(point) => point.y,
                ScrollDelta::Pixels(point) => f32::from(point.y) / 20.0,
            };
            handler(this, delta_y, window, cx);
        }))
    }
}

pub use inner::*;
