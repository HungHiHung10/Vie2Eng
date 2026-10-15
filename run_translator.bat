@echo off
title Google Dich - PhoMT AI Translator
echo =======================================================
echo          GOOGLE DICH - PHOMT AI TRANSLATOR
echo     MarianNMT INT8 + PhoMT Transformer (.pt)
echo =======================================================
echo Dang khoi dong may chu tai http://localhost:8000 ...
cd /d "%~dp0"
python server.py
pause
