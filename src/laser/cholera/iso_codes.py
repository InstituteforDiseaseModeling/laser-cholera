"""ISO-3 country codes for the 40 MOSAIC sub-Saharan-Africa countries covered by the model.

Sorted alphabetically, with the country name as a trailing comment on
each line. Used by
[`laser.cholera.metapop.scenario`][laser.cholera.metapop.scenario] to
filter the bundled demographics CSV down to the MOSAIC subset, and
exposed as `laser.cholera.iso_codes` for convenience in user code.
"""

iso_codes = sorted(
    [
        "AGO",  # Angola
        "BEN",  # Benin
        "BWA",  # Botswana
        "BFA",  # Burkina Faso
        "BDI",  # Burundi
        "CMR",  # Cameroon
        "CAF",  # Central African Republic
        "TCD",  # Chad
        "COG",  # Congo
        "COD",  # Democratic Republic of the Congo
        "CIV",  # Côte d'Ivoire
        "GNQ",  # Equatorial Guinea
        "ERI",  # Eritrea
        "SWZ",  # Eswatini
        "ETH",  # Ethiopia
        "GAB",  # Gabon
        "GMB",  # Gambia
        "GHA",  # Ghana
        "GIN",  # Guinea
        "GNB",  # Guinea-Bissau
        "KEN",  # Kenya
        "LSO",  # Lesotho
        "LBR",  # Liberia
        "MWI",  # Malawi
        "MLI",  # Mali
        "MRT",  # Mauritania
        "MOZ",  # Mozambique
        "NAM",  # Namibia
        "NER",  # Niger
        "NGA",  # Nigeria
        "RWA",  # Rwanda
        "SEN",  # Senegal
        "SLE",  # Sierra Leone
        "SOM",  # Somalia
        "ZAF",  # South Africa
        "SSD",  # South Sudan
        "TGO",  # Togo
        "UGA",  # Uganda
        "TZA",  # United Republic of Tanzania
        "ZMB",  # Zambia
        "ZWE",  # Zimbabwe
    ]
)
