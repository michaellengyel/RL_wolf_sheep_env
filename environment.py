import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import random
import cv2

class Environment:

    # Initalize environment and sets agent to random location (s0, a0)
    def __init__(self, map_img_path, fov, food_spawn_threshold, visuals=False):
        # Private input variables
        self.map_img_path = map_img_path
        self.fov = fov
        self.food_spawn_threshold = food_spawn_threshold
        #self.visuals = visuals

        # Private variables
        self.offset = int((self.fov - 1) / 2)
        self.map_img_agents = self.load_map()
        self.map_img_calculations = self.load_map()
        self.env_width = self.map_img_agents.shape[0]
        self.env_height = self.map_img_agents.shape[1]
        self.agent_x = 0
        self.agent_y = 0
        self.agent_reward = 0
        self.agent_current_reward = 0
        # Number of rewards randomly generated
        self.no_of_rewards = 0
        # Is the game done
        self.done = False

        # s0
        self.init_map()
        # a0
        self.init_agent_pos()


        cv2.namedWindow('MAP', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('MAP', 900, 900)
        cv2.namedWindow('SUBMAP', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('SUBMAP', 400, 400)


    ########## Environment Initialization Functions ##########

    # Initialize position of agent randomly (a0)
    def init_agent_pos(self):
        rand_x = random.randrange(self.offset, self.env_width - self.offset)
        rand_y = random.randrange(self.offset, self.env_height - self.offset)
        self.agent_x = rand_x
        self.agent_y = rand_y
        self.map_img_agents[self.agent_x, self.agent_y, 0] = 255
        #self.map_img[self.agent_x, self.agent_y, 1] = 0
        #self.map_img[self.agent_x, self.agent_y, 2] = 0

    # Load the map from a .png
    def load_map(self):
        img = plt.imread(self.map_img_path)
        map_img = img.copy()
        #map = torchvision.datasets.ImageFolder(root=map_path, transform=torchvision.transforms.ToTensor)
        return map_img

    # Spawn food
    def init_map(self):

        # Iterate through x dim
        for i in range(self.offset, self.env_width - self.offset):
            # Iterate through y dim
            for j in range(self.offset, self.env_height - self.offset):
                # Generate a number in the range of the pixel at i, j
                rand = random.randrange(0, self.map_img_agents[i, j, 0])
                # If the generated number is 0 and pixel i, j is sub 30
                if ((self.map_img_agents[i, j, 0] < self.food_spawn_threshold) & (rand == 0)):
                    # Set pixel red
                    self.map_img_agents[i, j, 0] = 0
                    self.map_img_agents[i, j, 1] = 255
                    self.map_img_agents[i, j, 2] = 0
                    # Increment the number of rewards counter
                    self.no_of_rewards += 1
                #print(self.map_img[i, j ,2])


    ########## Reset Function ##########

    def reset(self):
        self.map_img_agents = self.load_map()
        self.map_img_calculations = self.load_map()
        self.init_agent_pos()
        self.init_map()

        self.sub_map_img = self.map_img_agents[self.agent_x - self.offset:self.agent_x + self.offset + 1, self.agent_y - self.offset:self.agent_y + self.offset + 1, :]
        return self.sub_map_img

    ########## Getter Function ##########

    ########## Debug Helper Functions ##########

    # Might need to make a torch.tensor from the numpy array
    def make_tensor_from_image(self):
        print("Making torch.tensor from image")

    def print_map_info(self):
        print(type(self.map_img_agents))
        print(self.map_img_agents.shape)

    def print_agent_info(self):
        print(self.agent_y, self.agent_x)

    ########## GUI Render Functions ##########

    def render_map(self):
        cv2.imshow('MAP', self.map_img_agents)

    # Render the map of the environment each tick
    def render_sub_map(self):
        cv2.imshow('SUBMAP', self.sub_map_img)

    ########## Environment Logic Functions ##########

    def movement_decision(self, x, y):

        lower_inside_bounds_x = ((self.agent_x + x) > self.offset)
        lower_inside_bounds_y = ((self.agent_y + y) > self.offset)
        upper_inside_bounds_x = ((self.agent_x + x) < self.env_width - self.offset)
        upper_inside_bounds_y = ((self.agent_y + y) < self.env_height - self.offset)

        if ((lower_inside_bounds_x) & (lower_inside_bounds_y) & upper_inside_bounds_x & upper_inside_bounds_y):
            # Un-paint the agent
            self.map_img_agents[self.agent_x, self.agent_y, 0] = self.map_img_calculations[self.agent_x, self.agent_y, 0]
            self.map_img_agents[self.agent_x, self.agent_y, 1] = self.map_img_calculations[self.agent_x, self.agent_y, 1]
            self.map_img_agents[self.agent_x, self.agent_y, 2] = self.map_img_calculations[self.agent_x, self.agent_y, 2]
            # Update the agent's position
            self.agent_x += x
            self.agent_y += y
            # Paint the agent to it's new position
            self.map_img_agents[self.agent_x, self.agent_y, 0] = 255
            self.map_img_agents[self.agent_x, self.agent_y, 1] = 0
            self.map_img_agents[self.agent_x, self.agent_y, 2] = 0
        #else:
            #print("Out of bounds")

    # Calculating net reward
    def calculate_reward(self, x, y):
        # Logic for gaining rewards
        if ((self.map_img_agents[self.agent_x + x, self.agent_y + y, 1]) == 255):
            self.agent_reward += 30
            #print("Reward: ", self.agent_reward)
        else:
            self.agent_reward -= 1

    # Calculating step's reward
    def calculate_current_reward(self, x, y):
        # Logic for gaining rewards
        # Reset current reward to 0
        self.agent_current_reward = 0
        # If stepped on reward set current reward to 1
        if ((self.map_img_agents[self.agent_x + x, self.agent_y + y, 1]) == 255):
            self.agent_current_reward = 30
            #print("Reward: ", self.agent_current_reward)
        else:
            self.agent_reward -= 1

    # Update the environment based on the action
    def step(self, action):

        # UP
        if (action == 0):
            #print("UP")
            self.calculate_reward(0, -1)
            self.calculate_current_reward(0, -1)
            self.movement_decision(0, -1)
        # DOWN
        elif (action == 1):
            #print("DOWN")
            self.calculate_reward(0, 1)
            self.calculate_current_reward(0, 1)
            self.movement_decision(0, 1)
        # LEFT
        elif (action == 2):
            #print("LEFT")
            self.calculate_reward(-1, 0)
            self.calculate_current_reward(-1, 0)
            self.movement_decision(-1, 0)
        # RIGHT
        elif (action == 3):
            #print("RIGHT")
            self.calculate_reward(1, 0)
            self.calculate_current_reward(1, 0)
            self.movement_decision(1, 0)
        # UP/LEFT
        elif (action == 4):
            #print("UP/LEFT")
            self.calculate_reward(-1, -1)
            self.calculate_current_reward(-1, -1)
            self.movement_decision(-1, -1)
        # UP/RIGHT
        elif (action == 5):
            #print("UP/RIGHT")
            self.calculate_reward(1, -1)
            self.calculate_current_reward(1, -1)
            self.movement_decision(1, -1)
        # DOWN/LEFT
        elif (action == 6):
            #print("DOWN/LEFT")
            self.calculate_reward(-1, 1)
            self.calculate_current_reward(-1, 1)
            self.movement_decision(-1, 1)
        # DOWN/RIGHT
        elif (action == 7):
            #print("DOWN/RIGHT")
            self.calculate_reward(1, 1)
            self.calculate_current_reward(1, 1)
            self.movement_decision(1, 1)

        # Check if all rewards have been found, if yes set done to true

        if(self.no_of_rewards == self.agent_reward):
            self.done = True

        # Crop the submap from the map
        self.sub_map_img = self.map_img_agents[self.agent_x - self.offset:self.agent_x + self.offset + 1, self.agent_y - self.offset:self.agent_y + self.offset + 1, :]

        return self.sub_map_img, self.agent_current_reward, self.done


