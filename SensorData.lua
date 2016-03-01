--[[
DeepTracking: Seeing Beyond Seeing Using Recurrent Neural Networks.
Copyright (C) 2016  Peter Ondruska, Mobile Robotics Group, University of Oxford
email:   ondruska@robots.ox.ac.uk.
webpage: http://mrg.robots.ox.ac.uk/

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
--]]

cmd:option('-grid_minX', -25, 'occupancy grid bounds [m]')
cmd:option('-grid_maxX',  25, 'occupancy grid bounds [m]')
cmd:option('-grid_minY', -45, 'occupancy grid bounds [m]')
cmd:option('-grid_maxY',   5, 'occupancy grid bounds [m]')
cmd:option('-grid_step', 1, 'resolution of the occupancy grid [m]')
cmd:option('-sensor_start', -180, 'first depth measurement [degrees]')
cmd:option('-sensor_step', 0.5, 'resolution of depth measurements [degrees]')

SensorData = {}

function SensorData.__len(self)
	return { self.data:size(1), 2, self.height, self.width }
end

function SensorData.__index(self, i)
	local dist = self.data[i]:index(1, self.index):reshape(self.height, self.width)
	local input = torch.FloatTensor(2, self.height, self.width)
	input[1] = torch.lt(torch.abs(dist - self.dist), params.grid_step * 0.7071)
	input[2] = torch.gt(dist + params.grid_step * 0.7071, self.dist)
	return input
end

function LoadSensorData(file, params)
	local self = {}
	-- load raw 1D depth sensor data
	self.data = torch.load(file)
	self.width  = (params.grid_maxX - params.grid_minX) / params.grid_step + 1
	self.height = (params.grid_maxY - params.grid_minY) / params.grid_step + 1
	-- pre-compute lookup arrays
	self.dist  = torch.FloatTensor(self.height, self.width)
	self.index = torch.LongTensor(self.height, self.width)
	for y = 1,self.height do
		for x = 1,self.width do
			local px = (x - 1) * params.grid_step + params.grid_minX
			local py = (y - 1) * params.grid_step + params.grid_minY
			local angle = math.deg(math.atan2(px, py))
			self.dist[y][x]  = math.sqrt(px * px + py * py)
			self.index[y][x] = math.floor((angle - params.sensor_start) / params.sensor_step + 1.5)
		end
	end
	self.index = self.index:reshape(self.width * self.height)
	setmetatable(self, SensorData)
	return self
end
