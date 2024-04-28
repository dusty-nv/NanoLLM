#!/usr/bin/env python3
from tabulate import tabulate
from termcolor import cprint, colored


def print_table(rows, header=None, footer=None, color='green', attrs=None):
    """
    Print a table from a list[list] of rows/columns, or a 2-column dict 
    where the keys are column 1, and the values are column 2.
    
    Header is a list of columns or rows that are inserted at the top.
    Footer is a list of columns or rows that are added to the end.
    
    color names and style attributes are from termcolor library:
      https://github.com/termcolor/termcolor#text-properties
    """
    def limit_str(text, max_len):
        if not isinstance(text, str):
            return text
            
        if len(text) > max_len:
            return text[:max_len]
        else:
            return text
            
    if isinstance(rows, dict):
        rows = [[limit_str(key,35), limit_str(value, 75)] for key, value in rows.items() if not isinstance(value, dict)]    

    if header:
        if not isinstance(header[0], list):
            header = [header]
        rows = header + rows
        
    if footer:
        if not isinstance(footer[0], list):
            footer = [footer]
        rows = rows + footer
        
    cprint(tabulate(rows, tablefmt='simple_grid', numalign='center'), color, attrs=attrs)
