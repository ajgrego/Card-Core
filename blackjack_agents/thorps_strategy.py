"""
Thorp's Optimal Basic Strategy for Blackjack
-----------------------------------------------------------------------------------------------
This implements the optimal basic strategy for Blackjack as originally
developed by Edward O. Thorp in his book,  "Beat the Dealer" (1962).

Reference:
    Thorp, E. O. (1962). Beat the Dealer: A Winning Strategy for the Game of 
    Twenty-One. Random House, New York.

 Notes:
    - These tables represent optimal play for standard casino rules
    - Assumes a dealer stands on soft 17, 8 decks, late surrender allowed
    
"""

#actions
H = 'H' #Hit
S = 'S' #Stand
D = 'D' #Double (hit if not allowed)
P = 'P' #Split
R = 'R' #Surrender (hit if not allowed)

#Hard totals (player total 4-21 vs dealer upcard 2-11)
thorp_hard= {
    4: ["H","H","H","H","H","H","H","H","H","H"],
    5: ["H","H","H","H","H","H","H","H","H","H"],
    6: ["H","H","H","H","H","H","H","H","H","H"],
    7: ["H","H","H","H","H","H","H","H","H","H"],
    8: ["H","H","H","H","H","H","H","H","H","H"],
    9: ["H","D","D","D","D","H","H","H","H","H"],
    10: ["D","D","D","D","D","D","D","D","H","H"],
    11: ["D","D","D","D","D","D","D","D","D","D"],
    12: ["H","H","S","S","S","H","H","H","H","H"],
    13: ["S","S","S","S","S","H","H","H","H","H"],
    14: ["S","S","S","S","S","H","H","H","H","H"],
    15: ["S","S","S","S","S","H","H","H","R","H"],
    16: ["S","S","S","S","S","H","H","R","R","R"],
    17: ["S","S","S","S","S","S","S","S","S","S"],
    18: ["S","S","S","S","S","S","S","S","S","S"],
    19: ["S","S","S","S","S","S","S","S","S","S"],
    20: ["S","S","S","S","S","S","S","S","S","S"],
    21: ["S","S","S","S","S","S","S","S","S","S"],
}

#Soft totals (player total 13-21 vs dealer upcard 2-11)
thorp_soft= {
    13: ["H","H","H","D","D","H","H","H","H","H"],  
    14: ["H","H","H","D","D","H","H","H","H","H"],  
    15: ["H","H","D","D","D","H","H","H","H","H"],  
    16: ["H","H","D","D","D","H","H","H","H","H"],  
    17: ["H","D","D","D","D","H","H","H","H","H"],  
    18: ["S","D","D","D","D","S","S","H","H","H"],  
    19: ["S","S","S","S","S","S","S","S","S","S"],  
    20: ["S","S","S","S","S","S","S","S","S","S"],  
    21: ["S","S","S","S","S","S","S","S","S","S"],  

}

#Pair splitting (player pair 2-11 vs dealer upcard 2-11)
#Y= Split, N=dont  split
thorp_split= {
    2: ['Y','Y','Y','Y','Y','Y','N','N','N','N'],
    3: ['Y','Y','Y','Y','Y','Y','N','N','N','N'],
    4: ['N','N','N','Y','Y','N','N','N','N','N'],
    5: ['N','N','N','N','N','N','N','N','N','N'],  
    6: ['Y','Y','Y','Y','Y','N','N','N','N','N'],
    7: ['Y','Y','Y','Y','Y','Y','N','N','N','N'],
    8: ['Y','Y','Y','Y','Y','Y','Y','Y','Y','Y'],
    9: ['Y','Y','Y','Y','Y','N','Y','Y','N','N'],
    10: ['N','N','N','N','N','N','N','N','N','N'],
    11: ['Y','Y','Y','Y','Y','Y','Y','Y','Y','Y'], 

}

def get_thorp_action(player_total, is_soft, dealer_up, pair_value=None, can_double=True, can_split=False, can_surrender=True):
    """
    Get Thorp's recommended action for a given state.
    Args:
        player_total: Sum of player's cards
        is_soft: True if hand has usable Ace
        dealer_up: Dealer's upcard (2-11; 11=Ace)
        can_double: if doubling is allowed
        can_split: if this is a splittable pair
        can_surrender: if surrender is allowed
    Returns:
        Action: 'H', 'S', 'D', 'P', or 'R'
    """
    if dealer_up < 2 or dealer_up > 11:
        raise ValueError("dealer_up must be between 2 and 11")
    dealer_index = dealer_up - 2  # convert 2..11 -> 0..9
    if player_total > 21:
        return S
    if player_total < 4:
        return H
    # check if pairs can split first 
    if pair_value is not None and can_split and pair_value in thorp_split:
        if thorp_split[pair_value][dealer_index] =='Y':
            return P
    #soft actions
    if is_soft and player_total in thorp_soft:
        action = thorp_soft[player_total][dealer_index]
    else:
        action = thorp_hard.get(player_total, [S]*10)[dealer_index]
    #fallback to allowed actions
    if action == D and not can_double:
        return H
    if action == R and not can_surrender:
        return H
    return action

def compare_action_to_thorp(player_total, is_soft, dealer_up, actual_action, pair_value=None):
    """Check if a real action matches Thorp's recommendation.
            Returns:
                - matches
                - thorp_action
                - normalized_actual (for misinputs from user;makes testing easier)
    """
    thorp_action = get_thorp_action(player_total, is_soft, dealer_up, pair_value=pair_value)
    normalized = (actual_action or '')[0].upper() if actual_action else 'S'
    if normalized == 'Y':
        normalized = 'P'
    if normalized not in ('H','S','D','P','R'):
        normalized = normalized
    return (normalized == thorp_action, thorp_action, normalized)

def verify_rules():
    """
    Run comprehensive checks to ensure Thorp's strategy tables are correct.
    
    Returns:
        list: List of (test_name, passed) tuples
    """
    checks = []
    
    #Hard totals verification
    checks.append(("Double 9 vs 3-6", thorp_hard[9][1:5] == ['D','D','D','D']))
    checks.append(("Double 10 vs 2-9", thorp_hard[10][0:8] == ['D']*8))
    checks.append(("Double 11 vs 2-10", thorp_hard[11][0:9] == ['D']*9))
    checks.append(("Stand 12 vs 4-6", thorp_hard[12][2:5] == ['S','S','S']))
    checks.append(("Stand 13-16 vs dealer 2-6", all(thorp_hard[t][0:5] == ['S']*5 for t in (13,14,15,16))))
    checks.append(("Surrender 16 vs 9,10,A", thorp_hard[16][7:10] == ['R','R','R']))
    
    #soft totals verification
    checks.append(("Double soft 18 vs 3-6", thorp_soft[18][1:5] == ['D','D','D','D']))
    checks.append(("Hit soft 18 vs 9,10,A", thorp_soft[18][7:10] == ['H','H','H']))
    
    #Split verification
    checks.append(("Split A,A always", all(x=='Y' for x in thorp_split[11])))
    checks.append(("Split 8,8 always", all(x=='Y' for x in thorp_split[8])))
    checks.append(("Never split 5s", all(x=='N' for x in thorp_split[5])))
    checks.append(("Never split 10s", all(x=='N' for x in thorp_split[10])))
    
    return checks


def print_strategy_tables():
    #prints all strategy tables
    print("THORP'S BASIC STRATEGY TABLES")
    print("-"*70)
    
    print("\nHARD TOTALS:")
    print("Player | ", end="")
    for d in range(2, 12):
        print(f"{d:^4}", end="")
    print("\n" + "-"*60)
    
    for total in sorted(thorp_hard.keys()):
        print(f"  {total:2d}   | ", end="")
        for action in thorp_hard[total]:
            print(f"{action:^4}", end="")
        print()
    
    print("\nSOFT TOTALS:")
    print("Player | ", end="")
    for d in range(2, 12):
        print(f"{d:^4}", end="")
    print("\n" + "-"*60)
    
    for total in sorted(thorp_soft.keys()):
        print(f"A,{total-11:2d}   | ", end="")
        for action in thorp_soft[total]:
            print(f"{action:^4}", end="")
        print()
    
    print("\nPAIRS:")
    print("Player | ", end="")
    for d in range(2, 12):
        print(f"{d:^4}", end="")
    print("\n" + "-"*60)
    
    for pair in sorted(thorp_split.keys()):
        card_name = "A" if pair == 11 else str(pair)
        print(f"{card_name},{card_name}    | ", end="")
        for action in thorp_split[pair]:
            print(f"{action:^4}", end="")
        print()
    print("-"*70 + "\n")

if __name__ == "__main__":
    print("Thorp's Strategy Module - Verification Tests\n")
    
    # Run verification tests
    checks = verify_rules()
    print("Running strategy verification tests...\n")
    
    all_passed = True
    for test_name, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\nAll tests passed")
    else:
        print("\nSome tests failed.")
    
    # Print strategy tables
    print_strategy_tables()
    
    