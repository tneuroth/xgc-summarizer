#!/bin/bash
astyle --style=allman --indent=spaces=4 ./source/*pp
astyle --style=allman --indent=spaces=4 ./source/*h
astyle --style=allman --indent=spaces=4 ./source/*xx
rm ./source/*.orig

