# Card Clash
Welcome to Card Clash, a turn-based strategy card game where players battle against each other using unique cards and strategic gameplay.

## System requirements
* python 3.10
* pip3
* brew for Mac users
* scoop for Windows users

## Installation steps
### macOS
1. Install brew if not already installed using the following command in yout terminal:

        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

3. Extract the CardClashMac.zip file. Navigate to the folder in the terminal.
4. Run the following command to install Godot:

        brew install godot

5. Ensure the install.sh file is executable. Run:

       chmod 777 install.sh
6. Then, run the install.sh file with the following command. This will install any Python libraries your are
missing for the AI to work:

       ./install.sh

7. Finally, run the following command to launch the game. All commands before this need only be run once. After playing and closing the window, they just have to open the terminal and run the following command to start the game again.Before running the command, ensure the terminal current directory is CardClashMac:

       godot

* Note for mac users: If the game runs, but generated attack cards appear empty (no name, description or any integer attributes), move the backup_data.csv file to CardClashMac/AI/data/. The game should work now.

### Windows
1. Install scoop if not already installed. Refer to the following website for commands to install: https://scoop.sh
2. Extract the CardClashWin.zip file. Navigate to the folder in the terminal.
3. Run the following commands to install Godot:

       scoop bucket add extras
       scoop install extras/godot
4. Move the backup_data.csv file to CardClashWin/AI/data/
5. Finally, run the following command to launch the game. All commands before this need only be run once. After playing and closing the window, they just have to open the terminal and run the following command to start the game again. Before running the command, ensure the terminal current directory is CardClashWin:

       godot


## Game Rules
1. Card Generation:
During the first turn, known as the preliminary turn, each player is allowed to generate a maximum of 10 cards.
Cards are generated by clicking the red cards at the bottom left corner of the window.
After the preliminary turn, each player can only draw 5 cards in each subsequent turn.
At the end of each turn, any unused cards in the deck (cards that are neither main nor benched) will be destroyed.
2. Turn Actions:
During a turn, players have the flexibility to perform various actions:
    * Place Cards: Players can drag cards into one fo the 4 areas. Player 1's are is the bottom half of the arena, while Player 2 has the top half of the arena.
    * Attach Energy Cards: Players can attach as many energy cards as they want to any number of the 4 playable cards: the main card or the three benched cards. By doing this, they can prepare to launch an attack.
    * Swap Cards: Players have the option of swapping cards between their main and benched cards. By clicking the button 'Swap main and bench cards, they will be prompted to select which benched card they would like to swap with their main card. The cards are ordered bench 1 from the left to bench 3 to the rightmost.
    * Attack: Players can choose to attack, by clicking the "Fight" button. Given that they have sufficient energy, this action will reduce the HP of their enemy by the card's power. The description is also executed. The player's turn automatically ends after an attack.
    * Skip Turn: If desired, players can choose to skip their turn, by pressing the 'Skip Turn' button
3. Winning Conditions:
A player wins the game by successfully destroying all 4 cards of their opponent.

## Known Bugs
As mentioned several times, this game is experimental and is only intended as a simple platform to apply the developed AI. At this time there are several bugs within the game. This is however at an acceptable level, as the game can still be completed.
With that being said, there are a few things, that shouldn't be done to ensure a smooth experience. 
1. Do not place a card over another card which already placed.
2. Do not place cards into the opposite player's areas.
3. Only use the swap button when the main and the benched cards are filled. Unless, just drag cards from the deck into the desired area.