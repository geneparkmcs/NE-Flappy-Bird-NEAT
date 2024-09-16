import pygame
import random
import neat
import os

pygame.init()

SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Flappy Bird")

GRAVITY = 0.5
FLAP_STRENGTH = -10
PIPE_GAP = 150
PIPE_WIDTH = 70
PIPE_VELOCITY = -3

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

BIRD_IMG = pygame.Surface((40, 30))
BIRD_IMG.fill(RED)

birds = []
ge = []
nets = []

def remove(i):
    birds.pop(i)
    ge.pop(i)
    nets.pop(i)

class Bird:
    def __init__(self):
        self.x = 50
        self.y = SCREEN_HEIGHT // 2
        self.velocity = 0
        self.rect = pygame.Rect(self.x, self.y, 40, 30)

    def update(self):
        self.velocity += GRAVITY
        self.y += self.velocity
        self.rect.y = self.y

    def flap(self):
        self.velocity = FLAP_STRENGTH

    def draw(self, screen):
        screen.blit(BIRD_IMG, (self.x, self.y))

class Pipe:
    def __init__(self, x):
        self.x = x
        self.height = random.randint(100, SCREEN_HEIGHT - 100)
        self.top_rect = pygame.Rect(self.x, 0, PIPE_WIDTH, self.height)
        self.bottom_rect = pygame.Rect(self.x, self.height + PIPE_GAP, PIPE_WIDTH, SCREEN_HEIGHT)

    def update(self):
        self.x += PIPE_VELOCITY
        self.top_rect.x = self.x
        self.bottom_rect.x = self.x

    def draw(self, screen):
        pygame.draw.rect(screen, GREEN, self.top_rect)
        pygame.draw.rect(screen, GREEN, self.bottom_rect)

    def off_screen(self):
        return self.x < -PIPE_WIDTH

    def collides_with(self, bird):
        return self.top_rect.colliderect(bird.rect) or self.bottom_rect.colliderect(bird.rect)

def eval_genomes(genomes, config):
    global birds, ge, nets
    clock = pygame.time.Clock()
    pipes = [Pipe(SCREEN_WIDTH + 100)]
    score = 0

    for genome_id, genome in genomes:
        birds.append(Bird())
        ge.append(genome)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        genome.fitness = 0
  
    running = True
    while running and len(birds) > 0:
        clock.tick(600)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                quit()

        SCREEN.fill(WHITE)

        for pipe in pipes:
            pipe.update()
            pipe.draw(SCREEN)

        for i, bird in enumerate(birds):
            bird.update()
            bird.draw(SCREEN)

            if bird.y > SCREEN_HEIGHT or bird.y < 0:
                ge[i].fitness -= 1
                remove(i)

            for pipe in pipes:
                if pipe.collides_with(bird):
                    ge[i].fitness -= 1
                    remove(i)
                    
        for pipe in pipes[:]:
            if pipe.off_screen():
                pipes.remove(pipe)
                pipes.append(Pipe(SCREEN_WIDTH))
                score += 1

        for genome in ge:
            genome.fitness += 0.1

        for i, bird in enumerate(birds):
            output = nets[i].activate((bird.y, abs(bird.y - pipes[0].height), abs(bird.y - (pipes[0].height + PIPE_GAP))))
            if output[0] > 0.5:
                bird.flap()

        font = pygame.font.Font(None, 36)
        text = font.render(str(score), True, BLACK)
        SCREEN.blit(text, (SCREEN_WIDTH // 2, 50))
        pygame.display.flip()

    birds.clear()
    ge.clear()
    nets.clear()

def run(config_path):
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    pop.run(eval_genomes, 50)

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    run(config_path)