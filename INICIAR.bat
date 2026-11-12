@echo off
rem Inicia un servidor local y abre VigilEye en el navegador (la camara solo funciona en localhost o HTTPS).
cd /d "%~dp0"
echo.
echo  VigilEye -> http://localhost:8000
echo  Para cerrar el servidor, cierre esta ventana o pulse Ctrl+C.
echo.
start "" http://localhost:8000/index.html
where py >nul 2>nul
if %errorlevel%==0 (
  py -m http.server 8000
) else (
  python -m http.server 8000
)
pause
