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

local WeightedBCECriterion, parent = torch.class('WeightedBCECriterion', 'nn.Criterion')

local eps = 1e-12

function WeightedBCECriterion:updateOutput(input, target)
    local target, weights = target[1], target[2]
    -- - log(input) * target - log(1 - input) * (1 - target)
    self.buffer = self.buffer or torch.Tensor():typeAs(input)
    self.buffer:add(input, eps):log():cmul(weights)
    self.output = - torch.dot(target, self.buffer)
    self.buffer:mul(input, -1):add(1):add(eps):log():cmul(weights)
    self.output = (self.output - torch.sum(self.buffer) + torch.dot(target, self.buffer)) / input:nElement()
    return self.output
end

function WeightedBCECriterion:updateGradInput(input, target)
    local target, weights = target[1], target[2]
    -- - (target - input) / ( input (1 - input) )
    -- The gradient is slightly incorrect:
    -- It should have be divided by (input + eps) (1 - input + eps)
    -- but it is divided by input (1 - input + eps) + eps
    -- This modification requires less memory to be computed.
    self.buffer = self.buffer or torch.Tensor():typeAs(input)
    self.buffer:add(input, -1):add(-eps):cmul(input):add(-eps)
    self.gradInput:add(target, -1, input):cdiv(self.buffer):cmul(weights):div(target:nElement())
    return self.gradInput
end