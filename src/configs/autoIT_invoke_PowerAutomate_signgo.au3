; Mattia Tomasoni 2021.

; open Microsoft Power Automate Desktop
Run("C:\Program Files (x86)\Power Automate Desktop\PAD.Console.Host.exe")

; do not perform clicks based on absolute mouse coordinates (based on screen)
; do it, instead, on mouse coordinates based on a specific window
; Options -> Coord Mode -> Windows
AutoItSetOption('MouseCoordMode', 1)

; wait a little...
Sleep(15000) ;  this *is needed*, even if you later wait for the window to appear
Sleep(500000); 

; wait for Power Automate Desktop to appear
;WinWait('Power Automate')
; click on full screen button
; (if not present, this just click on an empty area at the top of the screen)
;WinActivate('Power Automate)') ; bring it to the front (again)
;MouseClick('primary', 1640, 90, 1, 0)
;MouseClick('left', 0, 0, 1, 0)
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; order workflows: oldest first (by clicking on "Modified" header)
;WinActivate('Power Automate') ; bring it to the front
;MouseClick('primary', 795, 192, 1, 0)
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; click on START FLOW BUTTON
;WinActivate('Power Automate') ; bring it to the front (again)
;821-224, 381-75 ICIICICICICICI
MouseClick('primary', 725, 290, 1, 0)
;Sleep(15000)
;MouseClick('primary', 725, 405, 1, 0)
;MouseClick('primary', 568, 237, 1, 0)

; wait a little...
Sleep(15000) ; the processing takes anyway half a minute

; close Power Automate Desktop... NOT NEEDED SINCE I KILL THE PROC FROM PYTHON
;WinActivate('Power Automate Desktop (Preview)')
;MouseClick('primary', 1898, 15, 1, 0)