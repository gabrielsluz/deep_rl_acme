from collections import deque
import numpy as np

class FrameStack:
    def __init__(self, frameStackDepth=10, stateShape = (3,3,2)):
        self.frame_stack = deque(maxlen=frameStackDepth)
        self.state_shape = stateShape
        self.stack_depth = frameStackDepth
        #self.frame_stack_shape = (self.state_shape[0],self.state_shape[1], self.stack_depth*self.state_shape[2])
        self.frame_stack_shape = (self.state_shape[0],self.state_shape[1], self.stack_depth)

    def clearStack(self):
        self.frame_stack.clear()

    def state(self, state):
        self.frame_stack.append(state)
        return self.assemblyState()

    def assemblyState(self):
        current_output_depth = self.stack_depth - 1
        #state_depth = self.state_shape[2]

        '''
        create blank stack from full shape
        '''
        state_stack = np.zeros(shape=self.frame_stack_shape)

        '''
        copy all available observations so far
        if there are not enough observations
        then, the rest of the stack will be
        filled with zeroes
        '''
        for state in self.frame_stack:

            '''
            for each shape depth, copy its contents
            to the end of the framestack
            '''
            #for current_state_depth in range(state_depth):
            for y in range(self.state_shape[0]):
                for x in range(self.state_shape[1]):
                    state_stack[y,x,current_output_depth] = state[y,x]#,current_state_depth]
            current_output_depth-=1
        
        # convert to list for predict compliance
        return state_stack.tolist()