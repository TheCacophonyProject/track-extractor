"""
classifier-pipeline - this is a server side component that manipulates cptv
files and to create a classification model of animals present
Copyright (C) 2018, The Cacophony Project

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
"""

from abc import ABC, abstractmethod


class DefaultConfig(ABC):
    @classmethod
    @abstractmethod
    def get_defaults(cls):
        """ The function to get default config. """
        ...

    @abstractmethod
    def validate(self):
        """ The function to get default config. """
        ...


def deep_copy_map_if_key_not_exist(from_map, to_map):
    for key in from_map:
        if isinstance(from_map[key], dict):
            if key not in to_map:
                to_map[key] = {}
            deep_copy_map_if_key_not_exist(from_map[key], to_map[key])
        elif key not in to_map:
            to_map[key] = from_map[key]
