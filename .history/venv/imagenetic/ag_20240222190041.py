import math
import numpy as np
import cv2 as cv



#Hiperparametros del modelo
POPULATION_SIZE = 50
EPOCHS = 10
CHROMOSOME_LENGTH = 5
#value vector example, later we can ask for parameter specification through console
VALUE_VECTOR = [1,2,3,4,5]
# constante para que el operador de cruza solo cambie un bit por default, y se el usuario
# quiere puede cambiar mas
BITS_MUTACION = 1
SURVIVOR_PERCENTAGE=0.1
PORCENTAJE_MUTACION=0.1
PORCENTAJE_REPRODUCCION=1

"""
    Representation: Each individual will represent a N x M 3 color channel image
    
    For this a 3d array numpy are can be used to store each value for each of the channel.
    
    Another thing is that we can represent pixels with values ranging from 0 to 255 or we can normalize values to be
    between 0 and 1
    
    In this case we will normalize the pixel values
    
    Also for the first version i am going to only allow grayscale images (black & white only)

"""


#image size in pixels
HEIGHT = 100
WIDTH = 100


#target image, hardcoded for now

input_image = cv.imread('input_image.png', cv.IMREAD_ANYCOLOR)
gray_image = cv.cvtColor(input_image, cv.COLOR_BGR2GRAY)
resized = cv.resize(gray_image, (WIDTH, HEIGHT))
caca = np.array(resized)
caca

def random_init(longitud_cromosoma, tam_poblacion):
    poblacion = list()
    for individuo in range(tam_poblacion):
        chromosome = list()
        for gene in range(longitud_cromosoma):
            initializer = np.random.rand()
            if initializer > 0.5:
                chromosome.append(1)
            else:
                chromosome.append(0)
        poblacion.append(chromosome)
    return poblacion


def fitness_func(cromosoma, value_vector=VALUE_VECTOR):
    input_image = cv.imread('input_image.png')
    target_image = cv.resize(input_image, (WIDTH, HEIGHT))
    target_pixel_values = np.array(target_image).flatten()
    
    evolved_image = np.array(cromosoma).flatten()
    
    mse = np.mean((target_pixel_values - evolved_image) ** 2)
    
    score = 0
    for i in range(len(cromosoma)):
        if cromosoma[i] != 0:
            score+=value_vector[i]

    return score
    
    
def selection_operator(survivor_percentage, poblacion):
    
    indivuals_w_scores = list()

    for individual in poblacion:
        score = fitness_func(individual)
        indivuals_w_scores.append((individual, score))
        
    sorted_individuals = sorted(indivuals_w_scores, key=lambda x: x[1], reverse=True)

    num_sobrevivientes = round(len(poblacion) * survivor_percentage)

    seleccionados = list()

    for i in range(num_sobrevivientes):
        seleccionados.append(sorted_individuals[i][0])

    return seleccionados
    
    
    
def one_point_crossover(elite, tam_pob):
    middle_point = round(len(elite)/2)
    
    left_half = elite[:middle_point]
    right_half = elite[middle_point:]
    
    hijos = list()
    for i in range(middle_point):
        random_point = np.random.randint(0,len(elite))
        papa = left_half[i]
        mama = right_half[i]
        
        hijo1 = (papa[:random_point] + mama[random_point:])
        hijo2 = (mama[:random_point] + papa[random_point:])
        
        hijos.append(hijo1)
        hijos.append(hijo2)
        
    ratio = round(tam_pob / len(hijos))
    
    new_gen = list()
    for i in range(len(hijos)):
        for _ in range(ratio):
            new_gen.append(hijos[i]) 
            
    return new_gen
    
def bit_flip_mutation(poblacion,porcentaje_mutacion=PORCENTAJE_MUTACION, bits=BITS_MUTACION):
    x_gen = list()
    total_mutados = 0
    for individuo in poblacion:
        loteria = np.random.randint(0,100)
        if loteria == 50:
            total_mutados+=1
            for _ in range(bits):
                random_bit = np.random.randint(0, len(individuo))
                if individuo[random_bit] == 0:
                    individuo[random_bit] = 1
                elif individuo[random_bit] == 1:
                    individuo[random_bit] = 0
                    
        x_gen.append(individuo)
        
    return x_gen, total_mutados
    
    
def genetic_algorithm(r_cross=1, r_mut=PORCENTAJE_MUTACION, total_genes=CHROMOSOME_LENGTH, population_size=POPULATION_SIZE,
                      value_vec=VALUE_VECTOR, epochs=EPOCHS, r_selection=SURVIVOR_PERCENTAGE):
    
    population = random_init(total_genes, population_size)
    
    for epoch in range(epochs):
        
        elite = selection_operator(r_selection, population)
        
        new_gen = one_point_crossover(elite, population_size)
        
        x_gen, total_mutados = bit_flip_mutation(new_gen, r_mut)
        
        population = x_gen
        
    performance = [fitness_func(individuo) for individuo in population]
    best_score = max(performance)
    index = performance.index(best_score)
    best = population[index]
    
    return best,best_score

best_solution, score = genetic_algorithm()
print("Done!")
print('f(%s) = %f' % (best_solution,score))
