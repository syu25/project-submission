@echo off
setlocal enabledelayedexpansion enableextensions
echo --------------------------------------------------------------------------------
echo This script normalizes images to 224x224 while maintaining aspect ratio and 
echo adding black padding where necessary for use with CUDA applications.
echo.
echo You will need the program ImageMagick and it will need to be registered to your 
echo system's PATH for the script to work.
echo.
echo GET IT HERE: https://imagemagick.org/script/download.php
echo.
echo This script assumes it is in a parent directory alongside folders containing the
echo extracted fish and mask sets.
echo.
echo For example, This should be in the same directory as the folders 
echo.
echo 	fish/fish_01,fish_02...
echo 	mask/mask_01,mask_02...
echo.
echo --------------------------------------------------------------------------------
pause
cls
echo --------------------------------------------------------------------------------
echo THIS SCRIPT WILL PERFORM A DESTRUCTIVE OPERATION ON ALL PNG FILES IN ADJACENT
echo DIRECTORIES AND THEIR CHILDREN. PROCEED WITH CAUTION.
echo.
echo ANY PNGS IN THE LISTED FOLDERS WILL BE MODIFIED^^!
echo READY TO OPERATE ON THE FOLLOWING DIRECTORIES:
echo.
tree
echo --------------------------------------------------------------------------------
SET /p input=CONFIRM OPERATION - THIS CANNOT BE UNDONE^^! (Y/N): 
IF /i '%input%'=='Y' GOTO EXEC 
GOTO ABORT
:EXEC
cls
FOR /R %%f IN (*.png) DO (
magick "%%f" -verbose -resize 224x224 -background black -gravity center -extent 224x224 -quality 100 "%%f"
)
cls
echo --------------------------------------------------------------------------------
echo Operation complete^^!
echo.
echo --------------------------------------------------------------------------------
GOTO END
:ABORT
cls
echo --------------------------------------------------------------------------------
echo Operation aborted^^!
echo.
echo --------------------------------------------------------------------------------
:END
pause
