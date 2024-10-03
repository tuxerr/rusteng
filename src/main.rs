use std::io::Read;
use std::{ffi::CStr, ffi::CString, u64};

use std::fs::File;
use std::path::Path;

use gltf;
use std::collections::HashMap;

use winit::{
    application::ApplicationHandler,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle},
    window::Window,
};

use ash_window;

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
        let win_attrib = Window::default_attributes().with_title("Foxy");
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
    // let (document, buffer, images) =
    //     gltf::import("assets/Fox.glb").expect("Unable to load Fox model");

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
 