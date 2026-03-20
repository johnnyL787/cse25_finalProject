import pygame
from random import randint
import time

SCREEN_SIZE = 800
BORDER_THICKNESS = 30

class Game:
    def __init__(self, grid_size=10):
        self.cell_size = SCREEN_SIZE // grid_size
        self.grid_size = grid_size
        self.board_size = self.cell_size * self.grid_size + BORDER_THICKNESS * 2

        pygame.init()
        self.screen = pygame.display.set_mode((self.board_size, self.board_size))
        self.clock = pygame.time.Clock()

        x = self.cell_size * (self.grid_size // 2) + BORDER_THICKNESS
        y = self.cell_size * (self.grid_size // 2) + BORDER_THICKNESS

        self.running = True
        self.snake = [(x, y), (x - self.cell_size, y), (x - 2 * self.cell_size, y)]
        self.score = 0
        self.direction = (1, 0)
        self.move = False
        self.prevScore = 0

    def mainloop(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            self.screen.fill("darkgreen")
            self._draw_board()
            self._draw_snake()
            self._draw_food()

            pygame.display.update()
            self.clock.tick(60)
        pygame.quit()

    def update(self, snake, food, reward) -> int:
        self.snake = snake
        self.food = food

        self.screen.fill("darkgreen")
        self._draw_board()
        self._draw_snake()
        self._draw_food()

        if reward == 10.0:
            self.score += 1

        pygame.display.update()
        self.clock.tick(60)
        return self.score

    def _draw_board(self):
        font = pygame.font.Font('freesansbold.ttf', 25)
        text = font.render('Score: ' + str(self.score), True, "black")
        textRect = text.get_rect(center=(self.board_size // 2, BORDER_THICKNESS / 2 + 5))
        self.screen.blit(text, textRect)

        for i in range(self.grid_size):
            y = i * self.cell_size + BORDER_THICKNESS
            if i%2 == 0: colors = ["green", (0, 150, 0)]
            else: colors = [(0, 150, 0), "green"]
            for j in range(self.grid_size):
                if j%2 == 0: color = colors[0]
                else: color = colors[1]
                pygame.draw.rect(self.screen, color, (j * self.cell_size + BORDER_THICKNESS, y, self.cell_size, self.cell_size))

    def _draw_snake(self):
        x = self.snake[0][0] * self.cell_size + BORDER_THICKNESS
        y = self.snake[0][1] * self.cell_size + BORDER_THICKNESS
        color = [0, 0, 0]
        gradient = 255 // len(self.snake)
        #pygame.draw.rect(self.screen, (0, 0, 200), (x, y, self.cell_size, self.cell_size))

        for i in range(len(self.snake)):
            x = self.snake[i][0] * self.cell_size + BORDER_THICKNESS
            y = self.snake[i][1] * self.cell_size + BORDER_THICKNESS
            pygame.draw.rect(self.screen, color, (x, y, self.cell_size, self.cell_size))
            color[2] += gradient

    def _draw_food(self):
        pygame.draw.rect(self.screen, "red", (self.food[0] * self.cell_size + BORDER_THICKNESS, self.food[1] * self.cell_size + BORDER_THICKNESS, self.cell_size, self.cell_size))