from threading import Thread
from time import sleep
from signal import signal, SIGINT

import pygame
import numpy as np


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


class DreamerDebugDisplay:
    def __init__(self, img_width: int, img_height: int, scaling: float=1):
        self.scaling = scaling
        self.img_width, self.img_height = img_width * scaling, img_height * scaling
        pygame.init()
        pygame.font.init()

        window_shape = (2 * self.img_width + 30, self.img_height + 50)
        self.screen = pygame.display.set_mode(window_shape, pygame.RESIZABLE)
        self.std_font = pygame.font.SysFont('Consolas', 14, bold=True)

        display_shape = (self.img_width, self.img_height)
        label_shape = (self.img_width, 30)
        self.display_orig = pygame.surface.Surface(display_shape)
        self.display_hall = pygame.surface.Surface(display_shape)
        self.label_orig = pygame.surface.Surface(label_shape)
        self.label_hall = pygame.surface.Surface(label_shape)

        def render_text(text: str) -> pygame.surface.Surface:
            return self.std_font.render(text, True, BLACK, WHITE)

        self.label_orig_text = render_text("original")
        self.label_hall_text = render_text("hallucinated")

        self.label_orig_offset = (10, 10)
        self.label_hall_offset = (self.img_width + 20, 10)
        self.display_orig_offset = (10, 40)
        self.display_hall_offset = (self.img_width + 20, 40)

        self.is_exit_requested = False

        def process_event_queue():
            while not self.is_exit_requested:
                for e in pygame.event.get():
                    if e.type == pygame.QUIT:
                        self.is_exit_requested = True
                sleep(0.01)

        self.ui_events_thread = Thread(target=process_event_queue)
        self.ui_events_thread.start()

        def handle_sigint(signum, frame):
            self.is_exit_requested = True

        signal(SIGINT, handle_sigint)

    def clear(self):
        self.screen.fill(WHITE)
        pygame.display.update()

    def next_frame(self, frame_orig: np.ndarray, frame_hall: np.ndarray):
        if self.is_exit_requested:
            self.ui_events_thread.join()
            pygame.quit()
            exit()

        self.screen.fill(WHITE)

        def show_text_centered(
                surface: pygame.surface.Surface,
                text: pygame.surface.Surface):
            total_width, total_height = surface.get_size()
            text_width, text_height = text.get_size()
            x_offset = total_width / 2 - text_width / 2
            y_offset = total_height / 2 - text_height / 2
            surface.fill(WHITE)
            surface.blit(text, (x_offset, y_offset))

        def prepare_frame(image: np.ndarray) -> np.ndarray:
            return np.rot90(np.fliplr(np.clip(image, 0, 255)))

        show_text_centered(self.label_orig, self.label_orig_text)
        show_text_centered(self.label_hall, self.label_hall_text)

        orig_surface = pygame.surfarray.make_surface(prepare_frame(frame_orig))
        hall_surface = pygame.surfarray.make_surface(prepare_frame(frame_hall))
        orig_surface = pygame.transform.scale(orig_surface, (self.img_width, self.img_height))
        hall_surface = pygame.transform.scale(hall_surface,  (self.img_width, self.img_height))
        self.display_orig.blit(orig_surface, (0, 0))
        self.display_hall.blit(hall_surface, (0, 0))

        self.screen.blit(self.display_orig, self.display_orig_offset)
        self.screen.blit(self.display_hall, self.display_hall_offset)
        self.screen.blit(self.label_orig, self.label_orig_offset)
        self.screen.blit(self.label_hall, self.label_hall_offset)

        pygame.display.update()
