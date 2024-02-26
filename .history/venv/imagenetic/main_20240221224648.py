import pygame
import numpy as np
from PIL import Image

# Define grid parameters
GRID_WIDTH = 50
GRID_HEIGHT = 50
CELL_SIZE = 10
WHITE = (255, 255, 255)

# Load input image and resize it to match grid size
input_image = Image.open('input_image.jpg')
input_image = input_image.resize((GRID_WIDTH, GRID_HEIGHT))
input_pixels = np.array(input_image)

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((GRID_WIDTH * CELL_SIZE, GRID_HEIGHT * CELL_SIZE))
pygame.display.set_caption("Population Visualization")

# Function to draw individual on the grid
def draw_individual(individual):
    for x in range(GRID_WIDTH):
        for y in range(GRID_HEIGHT):
            color = individual[x, y] * 255  # Scale pixel value to 0-255 for grayscale
            pygame.draw.rect(screen, (color, color, color), (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Generate random population (replace with your population data)
    population = np.random.randint(0, 2, size=(GRID_WIDTH, GRID_HEIGHT))

    # Clear screen
    screen.fill(WHITE)

    # Draw input image
    for x in range(GRID_WIDTH):
        for y in range(GRID_HEIGHT):
            color = input_pixels[x, y]
            pygame.draw.rect(screen, color, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    # Draw individuals in the population
    for individual_index in range(10):  # Draw only first 10 individuals for demonstration
        individual = population  # Replace with actual individual from your population
        draw_individual(individual)
    
    pygame.display.flip()

pygame.quit()
