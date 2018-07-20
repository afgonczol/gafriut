import numpy as np
import cv2 as cv
import random
from operator import itemgetter

#Read in image that will be recreated
input_img = cv.imread(r"C:\Users\c62wt96\.spyder-py3\gafriut\shapes800300.png")
orig_width = input_img.shape[1]
orig_height = input_img.shape[0]

#Reshape image to a max of 300 for performance reasons
ratio = 300 / max(orig_width, orig_height)
width = int(orig_width * ratio)
height = int(orig_height * ratio)
resized_input = cv.resize(input_img, (width, height))
#Change to 64bit integer so that formula for comparing images can work
resized_input64 = resized_input.astype(np.int64)

#Add extra room to drawing zone so that triangles aren't completely clustered around the center of the image.
draw_w_min = -width // 4
draw_h_min = -height // 4
draw_w_max = int(width * 1.25)
draw_h_max = int(height * 1.25)

#Inputs for running genetic algorith
n_genes = 10
n_pop = 1000
generations = 1000
verbose = 10
turns_worse_limit = 20 #Program will stop early if there are this many generations without further improvement

def create_image(triangles, colors):
    img = np.full((height, width, 3), 128, np.uint8)
    alpha = .7

    for triangle, color in zip(triangles, colors):
        overlay = img.copy()
        overlay = cv.fillPoly(overlay, [triangle], color)
        cv.addWeighted(img, alpha, overlay, 1-alpha, 0, img)
    return img

def create_triangle(width, height):
    draw_w_min = -width // 4
    draw_h_min = -height // 4
    draw_w_max = int(width * 1.25)
    draw_h_max = int(height * 1.25)
    
    return np.array([[[random.randrange(draw_w_min, draw_w_max), random.randrange(draw_h_min, draw_h_max)]], 
                           [[random.randrange(draw_w_min, draw_w_max), random.randrange(draw_h_min, draw_h_max)]], 
                           [[random.randrange(draw_w_min, draw_w_max), random.randrange(draw_h_min, draw_h_max)]]], np.int32)

def create_initial_pop(n_pop, n_genes, width, height, test_image):
    population = [0] * n_pop
    
    for i in range(n_pop):
        triangles = [0] * n_genes
        colors = [0] * n_genes
        for j in range(n_genes):
            triangles[j] = create_triangle(width, height)
            colors[j] = (random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255))
        img = create_image(triangles, colors)
        img64 = img.astype(np.int64)
        score = np.square(np.subtract(img64,test_image)).sum()
        population[i] = (triangles, colors, img, score)
        
    return sorted(population, key=itemgetter(3))

def create_next_pop(population, n_pop, width, height, orig_n_genes, test_image, generation):
    mutation_rate = .02 + .08 * (100/(100 + generation))
    parent_det = (1 - mutation_rate) / 2
    
    #Add more genees/triangles over time. Allows GA to focus on smaller number of triangles in the beginning and increase in complexity over time.
    n_genes = orig_n_genes + (generation // 20)
    
    tenth_percentile = int(n_pop * .1)
    next_pop = [0] * n_pop
    
    for n in range(n_pop):
        parent1 = random.randrange(0, tenth_percentile)
        parent2 = random.randrange(0, tenth_percentile)
        p1_triangles = population[parent1][0]
        p2_triangles = population[parent2][0]
        p1_colors = population[parent1][1]
        p2_colors = population[parent2][1]
        
        start_length = len(p1_triangles)
        triangles = [0] * n_genes
        colors = [0] * n_genes
        
        for i in range(n_genes):
            if i < start_length:
                rand = random.random()
                triangles[i] = p1_triangles[i] if rand < parent_det else p2_triangles[i] if rand < 2 * parent_det else create_triangle(width, height)
                colors[i] = p1_colors[i] if rand < parent_det else p2_colors[i] if rand < 2 * parent_det else (random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255))
            else:
                triangles[i] = create_triangle(width, height)
                colors[i] = (random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255))
        img = create_image(triangles, colors)
        img64 = img.astype(np.int64)
        score = np.square(np.subtract(img64,test_image)).sum()
        
        next_pop[n] = (triangles, colors, img, score)
        
    return sorted(next_pop, key=itemgetter(3))

if __name__ == '__main__':
    
    population = create_initial_pop(n_pop, n_genes, width, height, resized_input64)
    best_score = population[0][3]
    
    turns_worse = 0
    for g in range(1, generations + 1, 1):
        
        population = create_next_pop(population, n_pop, width, height, n_genes, resized_input64, g)
        current_score = population[0][3]
        if current_score < best_score:
            best_score = current_score
            best_img = population[0][2]
            best_generation = g
            turns_worse = 0
        else:
            turns_worse += 1
            if turns_worse >= turns_worse_limit:
                break
    
        if g % verbose == 0:
            print(f'Generation: {g}, Score: {current_score}')
    
    print(f'Best Generation: {best_generation}')
    print(f'Best Score: {best_score}')
    
        
    cv.imshow('Best Image',best_img)
    cv.waitKey(0)





















