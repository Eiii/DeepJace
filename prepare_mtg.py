#!/usr/bin/env python
import json
import sys
import operator

def main():
    set_filter = set()
    filename = sys.argv[1]
    fd = open(filename, 'r')
    for line in fd:
        info = json.loads(line.strip())

    sorted_keys = sorted(info, key=lambda key: info[key]["releaseDate"], reverse=True)

    for key in sorted_keys:
        """
        Name
        manaCost
        colors (?)
        types
        supertype
        subtype
        text
        ----
        power
        toughness
        loyalty
        """
        corpus = []
        for card in info[key]["cards"]:
            try:
                pass
                if card["type"] == "":
                    #checks for tokens
                    continue
                if "Land" in card["types"]:
                    #checks for lands
                    continue
                if "text" not in card:
                    #only get cards with rule text
                    continue
                else:
                    corpus.append(card["text"])
            except Exception as e:
                print >> sys.stderr, e
                print >> sys.stderr, card
        for line in corpus:
            try:
                print line
            except:
                pass
            
if __name__ == "__main__":
    main()
