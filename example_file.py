#!/usr/bin/python3
import melee

console = melee.Console(is_dolphin=False,
                        allow_old_version=False,
                        path="PATH_TO_SLP_FILE"
                        )
console.connect()

while True:
    gamestate = console.step()
    # step() returns None when the file ends
    if gamestate is None:
        break
    print("Frame " + str(gamestate.frame))
    for _, player in gamestate.player.items():
        print("\t", player.stock, player.percent)
