#!/bin/bash


GREEN="\033[92m"
YELLOW="\033[93m"
RED="\033[91m"
BLUE="\033[94m"
BOLD="\033[1m"
RESET="\033[0m"
NC='\033[0m' # No Color

INSTALLATION_LOGO_MAIN="
${BLUE}
___________________________________________
 __  __ ___ _   _ _____ ______     ___
|  \/  |_ _| \ | | ____|  _ \ \   / / \\
| |\/| || ||  \| |  _| | |_) \ \ / / _ \\
| |  | || || |\  | |___|  _<  \ V / ___ \\
|_|  |_|___|_| \_|_____|_| \_\ \_/_/   \_\\

    Benchmarks - Env Installation
___________________________________________
${NC}
"
DATASETS_LOGO_MAIN="
${BLUE}
___________________________________________
 __  __ ___ _   _ _____ ______     ___
|  \/  |_ _| \ | | ____|  _ \ \   / / \\
| |\/| || ||  \| |  _| | |_) \ \ / / _ \\
| |  | || || |\  | |___|  _<  \ V / ___ \\
|_|  |_|___|_| \_|_____|_| \_\ \_/_/   \_\\

    Benchmarks - Download Datasets
___________________________________________
${NC}
"
LOGO_SUCCESS="
${GREEN}
   ____  __
  / __ \/ /__
 / / / / //_/
/ /_/ / ,
\____/_/|_|   Installed successfully!
${NC}
"

LOGO_ERROR="
${RED}
 _____
| ____|_ __ _ __ ___  _ __
|  _| | '__| '__/ _ \| '__|
| |___| |  | | | (_) | |
|_____|_|  |_|  \___/|_|    Installtion failed!
${NC}
"

error_handler() {
    echo -e "$LOGO_ERROR"
    echo "❌ Error on line $1 during the execution of command: '$2'"
    echo "Exit code: $3"
    exit "$3"
}

# 2. Σύνδεση της συνάρτησης με το σήμα ERR
# Το $LINENO δείχνει τη γραμμή και το $BASH_COMMAND την εντολή που απέτυχε
trap 'error_handler $LINENO "$BASH_COMMAND" $?' ERR