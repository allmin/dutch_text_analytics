# Sample sentences For quick Validation
sentences = {"Meneer nam het medicijn in":["neemt medicatie in"], 
             "Meneer injecteerde het geneesmiddel zelf":["injectie van medicatie"]}
references = ["injectie van medicatie", "neemt medicatie in"]

demo_text= ["""Naam: [Naam van de Patiënt 1]
Datum: [Datum van Rapport]
Postoperatieve Status: De patiënt is stabiel en alert.
Wond: Geen infectie of zwelling.
Medicatie: [Naam van Medicatie 1] toegediend zonder allergische reacties.
Voeding: Vloeibaar dieet, goede inname.
Mobiliteit: Beperkte mobilisatie.
Afspraken: Volgende wondverbandwissel op [Datum en Tijd].
Opmerkingen: Familie geïnformeerd.
""",
"""Naam: [Naam van de Patiënt 2]
Datum: [Datum van Rapport]
Postoperatieve Status: Stabiel, matige pijn (4/10).
Wond: Geen tekenen van infectie.
Medicatie: [Naam van Medicatie 2] zonder bijwerkingen.
Voeding: Gestart met vast voedsel, matige inname.
Mobiliteit: Zittende positie bereikt.
Afspraken: Eerste verbandwissel gepland voor morgen.
Opmerkingen: Comfortabel, hulpvaardig.
"""]
