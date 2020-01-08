from environment import *

def main():

    env = Environment("data/reduced_height_map.jpg", 15, 50, True)
    no_action = 0
    counter = 0

    action = 0
    debug_mode_on = False

    # Simulate environment for development purposes
    while (True):
        #action = input()
        action = random.randrange(0, 8)

        action = int(action)

        next_state, reward, done = env.step(action)

        #env.step(int(action))
        env.render_map()
        env.render_sub_map()

        no_action += 1
        counter += 1
        if (counter >= 1000):
            counter = 0
            print("Action: ", no_action)

        # User Assist code
        if (debug_mode_on == True):
            k = cv2.waitKey(10000)
            if k == 97:  # a
                action = 0
            if k == 100:  # d
                action = 1
            if k == 119:  # w
                action = 2
            if k == 115:  # s
                action = 3
        elif(debug_mode_on == False):
            action = random.randrange(0, 8)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break



if __name__ == '__main__':
     main()