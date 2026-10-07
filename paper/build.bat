@echo off
setlocal
rem paper/build.bat - build paper/main.pdf deterministically.
rem Usage: paper\build.bat   (or from paper/: build.bat)
set "PATH=C:\Users\Vaibhav\AppData\Local\Programs\MiKTeX\miktex\bin\x64;%PATH%"
cd /d "%~dp0"

python make_numbers.py || goto :fail
pdflatex -interaction=nonstopmode -halt-on-error main.tex || goto :fail
bibtex main || goto :fail
pdflatex -interaction=nonstopmode -halt-on-error main.tex || goto :fail
pdflatex -interaction=nonstopmode -halt-on-error main.tex || goto :fail
echo BUILD OK: %~dp0main.pdf
exit /b 0

:fail
echo BUILD FAILED - see %~dp0main.log
exit /b 1
