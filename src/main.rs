use winit::{
    application::ApplicationHandler, event::WindowEvent, event_loop::{ControlFlow, EventLoop}, raw_window_handle::{HasDisplayHandle, RawDisplayHandle}, window::Window
};

#[cfg(target_os = "linux")]
use winit::platform::x11::EventLoopBuilderExtX11;


pub mod engine;

const APP_NAME: &str = "Foxy";

struct App {
    window: Option<Window>,
    engine: engine::Engine
}

impl App {
    fn new(disp_handle: RawDisplayHandle) -> Self {
        let engine = engine::Engine::new(disp_handle);
        App {
            window: None,
            engine: engine
        }
    }

}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let physical_size = winit::dpi::PhysicalSize { width : 1280, height : 1024 };
        let win_attrib = Window::default_attributes().with_title("Foxy").with_inner_size(physical_size);
        let win = event_loop.create_window(win_attrib);
        let actual_win = win.expect("Failure to create window");
        self.engine.window_init(&actual_win);
        self.window = Some(actual_win);
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                println!("The close button was pressed; stopping");
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                self.engine.render();
                self.window.as_ref().unwrap().request_redraw();
            }
            _ => (),
        }
    }
}

fn main() {
    println!("Engine start");

    #[cfg(target_os = "linux")]
    let event_loop = EventLoop::builder().with_x11().build().unwrap();
    
    #[cfg(target_os = "windows")]
    let event_loop = EventLoop::new().unwrap();

    event_loop.set_control_flow(ControlFlow::Poll);

    // predisplay handle for vulkan gfx init
    let disp_handle = event_loop
        .owned_display_handle()
        .display_handle()
        .expect("Failure to get predisplay")
        .as_raw();

    let mut app = App::new(disp_handle);

    event_loop
        .run_app(&mut app)
        .expect("Failure to run the app");
}
 