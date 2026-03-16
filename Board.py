import pygame
from random import randint
import time

SCREEN_SIZE = 800
BORDER_THICKNESS = 30

class Board:
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
        self.food = None
        self.direction = (1, 0)
        self.move = False

    def mainloop(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    self.move = True
                    if event.key == pygame.K_UP:
                        self._up()
                    elif event.key == pygame.K_DOWN:
                        self._down()
                    elif event.key == pygame.K_LEFT:
                        self._left()
                    elif event.key == pygame.K_RIGHT:
                        self._right()

            self.screen.fill("darkgreen")
            self._draw_board()
            self._render_smoothly(self.snake)
            self._spawn_food()
            self._update()
            self._check_collision()

            if self.snake[0] == self.food:
                self.food = None
                self._grow()

            time.sleep(0.1)
            pygame.display.update()
            self.clock.tick(60)
        pygame.quit()

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

    def _spawn_food(self):
        if self.food == None:
            self.food = (randint(0, self.grid_size - 1) * self.cell_size + BORDER_THICKNESS, randint(0, self.grid_size - 1) * self.cell_size + BORDER_THICKNESS)
            while self.food in self.snake:
                self.food = (randint(0, self.grid_size - 1) * self.cell_size + BORDER_THICKNESS, randint(0, self.grid_size - 1) * self.cell_size + BORDER_THICKNESS)
        pygame.draw.rect(self.screen, "red", (self.food[0], self.food[1], self.cell_size, self.cell_size))

    def _update(self):
        if not self.move: return

        new_head = (self.snake[0][0] + self.direction[0] * self.cell_size, 
                    self.snake[0][1] + self.direction[1] * self.cell_size)
        self.snake.insert(0, new_head)
        self.snake.pop()

    def _up(self):
        if self.direction != (0, 1):
            self.direction = (0, -1)

    def _down(self):
        if self.direction != (0, -1):
            self.direction = (0, 1)

    def _left(self):
        if self.direction != (1, 0):
            self.direction = (-1, 0)
    
    def _right(self):
        if self.direction != (-1, 0):
            self.direction = (1, 0)

    def _render_smoothly(self, prev_snake):
        for i in range(len(prev_snake)):
            if i == 0: color = (0, 0, 120)
            else: color = "blue"
            px, py = prev_snake[i]
            nx, ny = self.snake[i]

            x = px + (nx - px) * 1
            y = py + (ny - py) * 1

            pygame.draw.rect(self.screen, color, (x, y, self.cell_size, self.cell_size))

    def _grow(self):
        self.score += 1

        x = self.snake[-1][0] + -self.direction[0] * self.cell_size + BORDER_THICKNESS
        y = self.snake[-1][1] + -self.direction[1] * self.cell_size + BORDER_THICKNESS

        self.snake.append((x, y))

    def _check_collision(self):
        x = self.snake[0][0]
        y = self.snake[0][1]

        if x < BORDER_THICKNESS or x >= self.board_size - BORDER_THICKNESS or y < BORDER_THICKNESS or y >= self.board_size - BORDER_THICKNESS:
            self._death_screen()
        
        for pos in self.snake[1:]:
            if (x, y) == pos:
                self._death_screen()

    def _draw_snake(self):
        pygame.draw.rect(self.screen, "red", (self.snake[0][0], self.snake[0][1], self.cell_size, self.cell_size))
        for x, y in self.snake[1:]:
            pygame.draw.rect(self.screen, "blue", (x, y, self.cell_size, self.cell_size))


    def _death_screen(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            self.screen.fill("darkgreen")
            self._draw_board()
            self._draw_snake()
            pygame.display.update()
