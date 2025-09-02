; Mattia Tomasoni 2021.

; open Microsoft Power Automate Desktop
Run("C:\Program Files (x86)\Power Automate Desktop\PAD.Console.Host.exe")

; do not perform clicks based on absolute mouse coordinates (based on screen)
; do it, instead, on mouse coordinates based on a specific window
; Options -> Coord Mode -> Windows
AutoItSetOption('MouseCoordMode', 0)

; wait a little...
Sleep(3000) ;  this *is needed*, even if you later wait for the window to appear


; wait for Power Automate Desktop to appear
WinWait('Power Automate Desktop (Preview)')
WinActivate('Power Automate Desktop (Preview)') ; bring it to the front
; click on full screen button
; (if not present, this just click on an empty area at the top of the screen)
MouseClick('primary', 613, 16, 1, 0)
   ;MouseClick('primary', 895, 16, 1, 0)
   ;MouseClick('primary', 1124, 16, 1, 0)
; click on START FLOW BUTTON
WinActivate('Power Automate Desktop (Preview)') ; bring it to the front (again)
MouseClick('primary', 576, 245, 1, 0)
   ;MouseClick('primary', 568, 237, 1, 0)

; wait a little...
Sleep(15000) ; the processing takes anyway half a minute

; close Power Automate Desktop... NOT NEEDED SINCE I KILL THE PROC FROM PYTHON
;WinActivate('Power Automate Desktop (Preview)')
;MouseClick('primary', 1898, 15, 1, 0)