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

-- h0
function getInitialState(width, height)
    return torch.zeros(32, height, width)
end

-- One step of RNN --
-- {h0,x1} -> {h1,y1}
function getStepModule(width, height)
	local h0 = nn.Identity()()
	local x1 = nn.Identity()()

	local e  = nn.Sigmoid()( nn.SpatialConvolution(2, 16, 7, 7, 1, 1, 3, 3)(x1) )
	local j  = nn.JoinTable(1)({e,h0})
	local h1 = nn.Sigmoid()( nn.SpatialConvolution(48, 32, 7, 7, 1, 1, 3, 3)(j) )
	local y1 = nn.Sigmoid()( nn.SpatialConvolution(32, 1, 7, 7, 1, 1, 3, 3)(h1) )

	return nn.gModule({h0, x1}, {h1, y1})
end