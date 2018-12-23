
def flow2table(flow):
    table=[]
    for drop in flow:
        if not drop in table:
            table.append(drop)

    return table