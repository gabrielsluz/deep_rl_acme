from enum import Enum
class RewardFunctions(Enum):
    '''
    Simply define all possible 
    reward functions
    '''
    FOCAL_POINTS = 0
    PROJECTION = 1
    REACHING_PROJECTION = 2
    PROGRESS = 3
    PROGRESS_SHAPING = 4
