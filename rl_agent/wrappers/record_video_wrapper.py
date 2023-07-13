from acme import types
from acme.wrappers import base
import dm_env
import numpy as np
import os
import cv2

class RecordVideoWrapper(base.EnvironmentWrapper):
	"""
	Records videos of the environment. The environment must have a Push Simulator.
	"""	
	def __init__(self, environment: dm_env.Environment, video_dir:str, start_counter:int=0):
		super().__init__(environment)
		self._environment = environment	
		# Create video_dir if not exists
		if not os.path.exists(video_dir):
			os.makedirs(video_dir)
		self._video_dir = video_dir	
		# Inititalize video recording
		self._video_frames = []
		self.counter = start_counter

	def step(self, action) -> dm_env.TimeStep:
		# Get frame from simulator
		frame = self._environment.push_simulator.drawToBuffer()
		frame = (frame*255).astype(np.uint8)
		self._video_frames.append(frame)
		return self._environment.step(action)

	def reset(self) -> dm_env.TimeStep:
    	# Save video to mp4 format with 30 fps and file_name = video_{counter}.mp4
		if len(self._video_frames) > 0:
			out = cv2.VideoWriter(
    	    	os.path.join(self._video_dir, f'video_{self.counter}.mp4'),
    	    	cv2.VideoWriter_fourcc(*'mp4v'),
    	    	30,
    	    	(self._video_frames[0].shape[1], self._video_frames[0].shape[0]))
			for frame in self._video_frames:
				out.write(frame)
			out.release()
		self._video_frames = []
		self.counter += 1
		return self._environment.reset()