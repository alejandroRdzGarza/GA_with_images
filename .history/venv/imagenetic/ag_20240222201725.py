import math
import numpy as np
import cv2 as cv




"""
    Representation: Each individual will represent a N x M 3 color channel image
    
    For this a 3d array numpy are can be used to store each value for each of the channel.
    
    Another thing is that we can represent pixels with values ranging from 0 to 255 or we can normalize values to be
    between 0 and 1
    
    In this case we will normalize the pixel values
    
    Also for the first version i am going to only allow grayscale images (black & white only)

"""

#Hiperparametros del modelo
POPULATION_SIZE = 100
EPOCHS = 10
#image size in pixels
HEIGHT = 300
WIDTH = 300
CHROMOSOME_LENGTH = HEIGHT * WIDTH
#value vector example, later we can ask for parameter specification through console
VALUE_VECTOR = [1,2,3,4,5]
# constante para que el operador de cruza solo cambie un bit por default, y se el usuario
# quiere puede cambiar mas
BITS_MUTACION = CHROMOSOME_LENGTH
SURVIVOR_PERCENTAGE=0.1
PORCENTAJE_MUTACION=0.1
PORCENTAJE_REPRODUCCION=1



#target image, hardcoded for now

#load
input_image = cv.imread('input_image.png', cv.IMREAD_ANYCOLOR)
# convert to grayscale
gray_image = cv.cvtColor(input_image, cv.COLOR_BGR2GRAY)
#resize image
resized = cv.resize(gray_image, (WIDTH, HEIGHT))
#get pixel values
TARGET_IMAGE = np.array(resized)


def fitness_func(individuo, target_img=TARGET_IMAGE):
    mse = np.mean((target_img - individuo) ** 2)
    return mse
    
    
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
        
    selected = np.array(seleccionados)

    return selected
    
    
    
def one_point_crossover(elite, tam_pob):
    middle_point = round(len(elite)/2)
    
    left_half = elite[:middle_point]
    right_half = elite[middle_point:]
    
    hijos = list()
    for i in range(middle_point):
        papa = left_half[i]
        mama = right_half[i]
        assert papa.shape == mama.shape
        random_column = np.random.randint(0,len(papa))

        hijo1 = np.vstack((papa[:random_column], mama[random_column:]))
        hijo2 = np.vstack((mama[:random_column], mama[random_column:]))
      
        hijos.append(hijo1)
        hijos.append(hijo2)
        
    ratio = math.ceil(tam_pob / len(hijos))
    
    
    new_gen = list()
    for i in range(len(hijos)):
        for _ in range(ratio):
            new_gen.append(hijos[i]) 
            
    new_generation = np.array(new_gen)
            
    return new_generation
    
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
    
    population = np.random.rand(population_size, HEIGHT, WIDTH)
    
    for epoch in range(epochs):
        
        elite = selection_operator(r_selection, population)
        
        new_gen = one_point_crossover(elite, population_size)
        
        x_gen, total_mutados = bit_flip_mutation(new_gen, r_mut)
        
    #     population = x_gen
        
    # performance = [fitness_func(individuo) for individuo in population]
    # best_score = max(performance)
    # index = performance.index(best_score)
    # best = population[index]
    
    # return best,best_score
    return 0

# best_solution, score = genetic_algorithm()
# print("Done!")
# print('f(%s) = %f' % (best_solution,score))

genetic_algorithm()
