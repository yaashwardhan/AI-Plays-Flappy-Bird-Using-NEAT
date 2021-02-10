"""
Its Flappy Bird, But an AI plays it using Neuroevolution of Augmenting Topologies and Masked Pixel Precision Accuracy.
Programmer: Yashwardhan Deshmukh
github.com/yaashwardhan
yaashwardhan.me

Approach: Programming Flappy Bird Classes --> Implementing Physics and Death Mechanics --> Implementing Pixel Precision Collision Accuracy and Mechanics --> Implementing a Neural Network using N.E.A.T --> Breeding 14 Birds every generation until a certain fitness threshold (1000 here) is obtained
"""

import os
import neat
import time
# to randomize pipe generation
import random
# python module designed for writing video games
import pygame

pygame.font.init()
pygame.display.set_caption('Flappy Bird AI')
window_font = pygame.font.SysFont("arial", 18)
flappy_font = pygame.font.Font("assets/flappy-font.ttf", 50)
window_width = 500
window_height = 800
generations = 0
all_bird_imgs = [pygame.transform.scale2x(pygame.image.load(os.path.join("assets", "bird1.png"))), pygame.transform.scale2x(pygame.image.load(os.path.join(
    "assets", "bird2.png"))), pygame.transform.scale2x(pygame.image.load(os.path.join("assets", "bird3.png"))), pygame.transform.scale2x(pygame.image.load(os.path.join("assets", "bird4.png")))]
img_pipe = pygame.transform.scale2x(
    pygame.image.load(os.path.join("assets", "pipe.png")))
img_base = pygame.transform.scale2x(
    pygame.image.load(os.path.join("assets", "base.png")))
img_background = pygame.transform.scale2x(
    pygame.image.load(os.path.join("assets", "background.png")))


class Bird():
    animation_imgs = all_bird_imgs

    # while up down movement of the bird
    maximum_rotation = 24

    # velocity with which we rotate the bird
    rotation_velocity = 18

    # flap animation duration
    flap_animation = 8

    def __init__(self, x, y, rip):

        # x, y are the starting coordinates, rip (rest in peace) is the boolean value that checks if the bird is alive or not
        self.x = x
        self.y = y
        self.rip = rip
        self.tilt = 0
        self.ticks = 0
        self.vel = 0
        self.height = self.y
        self.img_count = 0
        self.img = self.animation_imgs[0]

    def rip_animation(self):
        # positive velocity will make it go in down as the y coordinate of the pygame window increases as we go down
        self.vel = 10
        self.ticks = 0
        # if bird is rip (by hitting a pipe) then it will turn the bird red and move it to the ground where we will remove it from the list of birds
        self.height = window_height

    def move(self):
        self.ticks = self.ticks + 1
        # d stands for displacement
        d = self.vel * (self.ticks) + 1.5 * self.ticks**2
        if d >= 14:
            d = 14
        if d < 0:
            d -= 2
        self.y = self.y + d

        if d < 0 or self.y < self.height + 50:
            if self.tilt < self.maximum_rotation:
                self.tilt = self.maximum_rotation
        else:
            if self.tilt > -90:
                self.tilt -= self.rotation_velocity

    def jump(self):
        # since top left corner of pygame window has coordinates (0,0), so to go upwards we need negative velocity
        self.vel = -10
        self.ticks = 0
        self.height = self.y

    def draw(self, win):
        # img_count will represent how many times have we already shown image
        self.img_count = self.img_count + 1

        # condition to check if the bird is alive
        if self.rip == False:
            # checking what image of the bird we should show based on the current image count
            if self.img_count <= self.flap_animation:
                self.img = self.animation_imgs[0]

            elif self.img_count <= self.flap_animation * 2:
                self.img = self.animation_imgs[1]

            elif self.img_count <= self.flap_animation * 3:
                self.img = self.animation_imgs[2]

            elif self.img_count <= self.flap_animation * 4:
                self.img = self.animation_imgs[1]

            elif self.img_count == self.flap_animation * 4 + 1:
                self.img = self.animation_imgs[0]
                self.img_count = 0

            # this will prevent flapping of the birds wings while going down
            if self.tilt <= -80:
                self.img = self.animation_imgs[1]
                self.img_count = self.flap_animation * 2

        # condition if the bird is rip
        elif self.rip == True:
            self.tilt = -90
            self.img = self.animation_imgs[3]
            self.rip_animation()

        # to rotate image in pygame
        rotated_image = pygame.transform.rotate(self.img, self.tilt)
        new_rect = rotated_image.get_rect(
            center=self.img.get_rect(topleft=(self.x, self.y)).center)
        # blit means draw, here we will draw the bird depending upon its tilt
        win.blit(rotated_image, new_rect.topleft)

    # since we want pixel perfect collision, and not just have a border around the bird, we mask the bird
    def get_mask(self):
        return pygame.mask.from_surface(self.img)


class Pipe:
    gap = 220
    # velocity of pipes, since the pipes move and the bird does not
    vel = 4

    def __init__(self, x):
        # x because the pipes are going to be random
        self.x = x
        self.height = 0
        # creating varibles to keep track of where the top and bottom of the pipe are going to be drawn
        self.top = 0
        self.bottom = 0
        self.top_pipe = pygame.transform.flip(img_pipe, False, True)
        self.bottom_pipe = img_pipe
        # if the bird has passed the pipe
        self.passed = False
        # this method will show where our pipes are and what the gap between them is
        self.set_height()

    def set_height(self):
        # randomizes the placement of pipes
        self.height = random.randrange(40, 450)
        self.top = self.height - self.top_pipe.get_height()
        self.bottom = self.height + self.gap

    def move(self):
        self.x = self.x - self.vel

    def draw(self, win):
        win.blit(self.top_pipe, (self.x, self.top))
        win.blit(self.bottom_pipe, (self.x, self.bottom))

    def collide(self, bird, win):
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.top_pipe)
        bottom_mask = pygame.mask.from_surface(self.bottom_pipe)

        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))
        # finding the point of collision
        bottom_point = bird_mask.overlap(bottom_mask, bottom_offset)
        top_point = bird_mask.overlap(top_mask, top_offset)

        if top_point or bottom_point:
            return True

        return False


class Base():
    vel = 5
    width = img_base.get_width()
    img = img_base

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.width

    def move(self):
        self.x1 -= self.vel
        self.x2 -= self.vel

        if self.x1 + self.width < 0:
            self.x1 = self.x2 + self.width

        if self.x2 + self.width < 0:
            self.x2 = self.x1 + self.width

    def draw(self, win):
        win.blit(self.img, (self.x1, self.y))
        win.blit(self.img, (self.x2, self.y))


def draw_window(win, birds, pipes, base, score, gen, pipe_number, fitness):
    if gen == 0:
        gen = 1
    win.blit(img_background, (0, 0))

    for pipe in pipes:
        pipe.draw(win)

    # showing all the text labels
    score_text = flappy_font.render(str(score), 0, (255, 255, 255))
    win.blit(score_text, (window_width - score_text.get_width() - 230, 100))
    # showing the generation number
    gen_text = window_font.render(
        "Species Generation Num: " + str(gen), 1, (0, 0, 0))
    win.blit(gen_text, (10, 5))
    # showing the number of birds that are alive in the provided frame
    alive_text = window_font.render("Alive: " + str(len(birds)), 1, (0, 0, 0))
    win.blit(alive_text, (10, 25))
    # showing the total number of birds that have been mutated in the current frame
    mutated_text = window_font.render(
        "Mutated: " + str(15 - len(birds)), 1, (231, 84, 128))
    win.blit(mutated_text, (10, 45))
    # showing the fitness value of the birds
    fitness_text = window_font.render(
        "Fitness: " + str(fitness), 1, (0, 255, 0))
    win.blit(fitness_text, (10, 65))
    # showing the fitness threshold that should be reached before automatically terminating the program
    fitness_t_text = window_font.render(
        "Fitness Threshold: 1000", 1, (0, 0, 0))
    win.blit(fitness_t_text, (window_width - fitness_t_text.get_width() - 10, 5))
    # showing the population of the birds that will be bred every generation
    population_text = window_font.render("Population: 15", 1, (0, 0, 0))
    win.blit(population_text, (window_width -
                               population_text.get_width() - 10, 25))

    base.draw(win)
    for bird in birds:
        bird.draw(win)
    pygame.display.update()
    

# the main() will also act as a fitness function for the program

def main(genomes, config):
    global generations
    generations += 1
    nets = []
    ge = []
    birds = []

    # setting neural network for genome, using _, g as 'genomes' is a tuple that has the genome id as well as the genome object
    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        birds.append(Bird(210, 320, rip=False))
        g.fitness = 0
        ge.append(g)

    base = Base(730)
    # first pipe will be a little away that other pipes so the birds knows that they can gain fitness by staying alive
    pipes = [Pipe(600)]

    win = pygame.display.set_mode((window_width, window_height))
    clock = pygame.time.Clock()

    score = 0

    run = True
    while run and len(birds) > 0:
        clock.tick(60)
        # keeps track if something happens like whenever user clicks keys etc.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                # quits the program when user hits the cross
                pygame.quit()
                quit()

        # making sure the bird looks only at first pipe and not the second pipe, since there can be multiple pipes generated
        pipe_number = 0
        if len(birds) > 0:
             # if we pass the first pipe, then look at the next pipe
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].top_pipe.get_width():
                pipe_number = 1

        for bird in birds:
            bird.move()
            # adding fitness since it has come to this level, also giving such little fitness value because the for loop will run 60 times a second so every second our bird stays alive, it will give it some fitness point, so this encourages the bird to stay alive
            ge[birds.index(bird)].fitness += 0.2

            output = nets[birds.index(bird)].activate((bird.y, abs(
                bird.y - pipes[pipe_number].height), abs(bird.y - pipes[pipe_number].bottom)))
            # if output of the neural network is > 0, we will make the bird jump ( > 0 was figured out by trial and error)
            if output[0] > 0:
                bird.jump()
        base.move()
        # list to remove pipes
        add_pipe = False
        rem = []
        for pipe in pipes:
            pipe.move()
            # checking for collision
            for bird in birds:
                if pipe.collide(bird, win):
                    # every time a bird hits a pipe, we will reduce its score, so we are encouraging the bird to go between the pipes
                    ge[birds.index(bird)].fitness -= 1
                    birds[birds.index(bird)].rip = True

                    # no need to add the below lines to delete the bird as, on hitting a pipe, we play the rip animation which turns the bird red and moves it to the ground and we already will pop the bird if it hits the ground so we dont need this.
                    # nets.pop(birds.index(bird))
                    # ge.pop((birds.index(bird)))
                    # birds.pop(birds.index(bird))

            # this is checking if our pipe is off the screen so we can generate another pipe
            if pipe.x + pipe.top_pipe.get_width() < 0:
                rem.append(pipe)

            if not pipe.passed and pipe.x < bird.x:
                pipe.passed = True
                add_pipe = True

        if add_pipe:
            score += 1
            for g in ge:
                g.fitness += 4
            # distance between pipes
            pipes.append(Pipe(550))

        for r in rem:
            pipes.remove(r)

       # checks if the bird hits the ground or goes above the screen and avoids the pipe
        for bird in birds:
            if bird.y + bird.img.get_height() - 10 >= 730 or bird.y < -50:
                # mutate the bird if it hits the ground
                nets.pop(birds.index(bird))
                ge.pop((birds.index(bird)))
                birds.pop(birds.index(bird))

        draw_window(win, birds, pipes, base, score,
                    generations, pipe_number, g.fitness)


def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    p = neat.Population(config)
    # now we will add stats reporters which will give us some outputs in the console where we will see the detailed statistics of each generation and the fitness etc.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(main, 100)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "assets/flappy-config.txt")
    run(config_path)


# Thank you! If you liked this project, you could show me support by following me or starring my repos --> github.com/yaashwardhan
