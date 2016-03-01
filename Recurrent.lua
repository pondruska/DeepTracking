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

-- N steps of RNN
-- {x1, x2, ..., xN, h0} -> {y1, y2, ..., yN}
--[[
-- A classical implementation of the recurrent module; requires O(N * M(step)) memory.
function Recurrent(step, N)
   local h = nn.Identity()()
   local hx, y = { [params.N+1] = h }, {}
   for i=1,N do
      hx[i] = nn.Identity()()
      local hy = step:clone('weight', 'bias', 'gradWeight', 'gradBias')({h, hx[i]})
      h    = nn.SelectTable(1)(hy)
      y[i] = nn.SelectTable(2)(hy)
   end
   return nn.gModule(hx, y)
end
--]]

-- A memory-efficient (but slower) version of the recurrent module; requires O(N * M(h0)) memory.
local Recurrent, parent = torch.class('Recurrent', 'nn.Module')

function Recurrent:__init(step, N)
   parent.__init(self)
   self.N = N
   self.step = step
   self.hidden    = {}
   self.output    = {}
   self.gradInput = {}
   for i=1,N do
      self.hidden[i] = torch.Tensor()
      self.output[i] = torch.Tensor()
      self.gradInput[i] = torch.Tensor()
   end
   self.hidden[0] = torch.Tensor()
   self.gradInput[N+1] = torch.Tensor()
end

-- resize A to size of B and sets it to 0
function zero(A,B)
   A:resizeAs(B):zero()
end

-- copy B to A
function copy(A,B)
   A:resizeAs(B):copy(B)
end

function Recurrent:forward(input)
   copy(self.hidden[0], input[self.N+1])
   for i = 1,self.N do
      local stepInput  = {self.hidden[i-1], input[i]}
      local stepOutput = self.step:forward(stepInput)
      copy(self.hidden[i], stepOutput[1])
      copy(self.output[i], stepOutput[2])
   end
   return self.output
end

function Recurrent:backward(input, gradOutput)
   zero(self.gradInput[self.N+1], input[self.N+1])
   for i = self.N,1,-1 do
      local stepInput  = {self.hidden[i-1], input[i]}
      local stepOutput = self.step:forward(stepInput)
      local stepGradOutput = {self.gradInput[self.N+1], gradOutput[i]}
      local stepGradInput  = self.step:backward(stepInput, stepGradOutput)
      copy(self.gradInput[self.N+1], stepGradInput[1])
      copy(self.gradInput[i], stepGradInput[2])
   end
   return self.gradInput
end
