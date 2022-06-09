# Africa, Asia-Temperate, Asia-Tropical, Australasia, Europe, Northern America, Pacific, Southern America
# World Geographical Scheme for Recording Plant Distributions, Brummitt
Africa = "Africa"
Antarctic = "Antarctic"
ArabPeninsula = "Asia-Temperate"
AsiaTemperate = "Asia-Temperate"
AsiaTropical = "Asia-Tropical"
Australasia = "Australasia"
Caribbean = "Caribbean"
CentralAmerica = "Northern America"
Caucasus = "Asia-Temperate"
Europe = "Europe"
NorthAmerica = "Northern America"
Pacific = "Pacific"
SouthAmerica = "Southern America"
WesternAsia = "Asia-Temperate"


country_code_dict = {"AX": Europe,          # Aaland Islands
                     "AF": WesternAsia,     # Afghanistan
                     "AL": Europe,          # Albania
                     "DZ": Africa,          # Algeria
                     "AS": Pacific,         # American Samoa
                     "AD": Europe,          # Andorra
                     "AO": Africa,          # Angola
                     "AI": Caribbean,       # Anguilla
                     "AQ": Antarctic,       # Antarctic
                     "AG": Caribbean,       # Antigua
                     "AR": SouthAmerica,    # Argentina
                     "AM": Caucasus,        # Armenia
                     "AW": Caribbean,       # Caribbean
                     "AU": Australasia,     # Australia
                     "AT": Europe,          # Austria
                     "AZ": Caucasus,        # Azerbaijan
                     "BS": Caribbean,       # Bahamas
                     "BH": ArabPeninsula,   # Bahrain
                     "BD": AsiaTropical,    # Bangladesh
                     "BB": Caribbean,       # Barbados
                     "BY": Europe,          # Belarus
                     "BE": Europe,          # Belgium
                     "BZ": Caribbean,       # Belize
                     "BJ": Africa,          # Benin
                     "BM": Caribbean,       # Bermuda
                     "BT": AsiaTropical,    # Bhutan
                     "BO": SouthAmerica,    # Bolivia
                     "BQ": Caribbean,       # Bonaire
                     "BA": Europe,          # Bosnia
                     "BW": Africa,          # Botswana
                     "BV": Antarctic,       # Bouvet Island
                     "BR": SouthAmerica,    # Brazil
                     "IO": Africa,          # British Indian Ocean Terrority -> Western Indian Ocean -> Africa
                     "BN": AsiaTropical,    # Brunei -> Malesia -> AsiaTropical
                     "BG": Europe,          # Bulgaria
                     "BF": Africa,          # Burkina Faso
                     "BI": Africa,          # Burundi
                     "CV": Africa,          # Cape Verdi
                     "KH": AsiaTropical,    # Cambodia -> Indo-China -> AsiaTropical
                     "CM": Africa,          # Cameroon
                     "CA": NorthAmerica,    # Canada
                     "KY": Caribbean,       # Cayman Islands
                     "CF": Africa,          # Central African Republic
                     "TD": Africa,          # Chad
                     "CL": SouthAmerica,    # Chile
                     "CN": AsiaTemperate,   # China
                     "CX": AsiaTropical,    # Christmas Island -> Malesia
                     "CC": Caribbean,       # Cocos
                     "CO": SouthAmerica,    # Colombia
                     "KM": Africa,          # Comoros -> West Indian Ocean
                     "CD": Africa,          # DRC
                     "CG": Africa,          # Republic of Congo
                     "CK": Pacific,         # Cook Islands
                     "CR": CentralAmerica,  # Costa Rica
                     "HR": Europe,           # Croatia
                     "CU": Caribbean,       # Cuba
                     "CW": Caribbean,       # Curacao
                     "CY": Europe,          # Cyprus
                     "CZ": Europe,          # Czech
                     "CI": Africa,          # Ivory coast
                     "DK": Europe,          # Denmark
                     "DJ": Africa,          # Djibouti
                     "DM": Caribbean,       # Dominica
                     "DO": Caribbean,       # Domincan Republic
                     "EC": SouthAmerica,    # Ecuador
                     "EG": Africa,          # Egypt
                     "SV": CentralAmerica,  # El Salvador
                     "GQ": Africa,          # Equatorial Guinea
                     "ER": Africa,          # Eritrea
                     "EE": Europe,          # Estonia
                     "SZ": Africa,          # Eswatini
                     "ET": Africa,          # Ethiopia
                     "FK": Antarctic,       # Falklands
                     "FO": Europe,          # Faroes
                     "FJ": Pacific,         # Fiji
                     "FI": Europe,          # Finland
                     "FR": Europe,          # France
                     "GF": SouthAmerica,    # French Guiana
                     "PF": Pacific,         # FRench Polynesia
                     "TF": Antarctic,       # French Southern Territories
                     "GA": Africa,          # Gabon
                     "GM": Africa,          # Gambia
                     "GE": Caucasus,        # Georgia
                     "DE": Europe,          # Germany
                     "GH": Africa,          # Ghana
                     "GI": Europe,          # Gibraltar
                     "GR": Europe,          # Greece
                     "GL": NorthAmerica,    # Greenland
                     "GD": Caribbean,       # Grenada
                     "GP": Caribbean,       # Guadeloupe
                     "GU": Pacific,         # Guam
                     "GT": CentralAmerica,  # Guatemala
                     "GG": Europe,          # Guernsey
                     "GN": Africa,          # Guinea
                     "GW": Africa,          # Guinea-Bissau
                     "GY": SouthAmerica,    # Guyana
                     "HT": Caribbean,       # Haiti
                     "HM": Antarctic,       # Heard Islands
                     "VA": Europe,          # Vatican
                     "HN": CentralAmerica,  # Honduras
                     "HK": AsiaTemperate,   # Hong Kong
                     "HU": Europe,          # Hungary
                     "IS": Europe,          # Iceland
                     "IN": AsiaTropical,    # India
                     "ID": AsiaTropical,    # Indonesia
                     "IR": AsiaTemperate,   # Iran
                     "IQ": AsiaTemperate,   # Iraq
                     "IE": Europe,          # Ireland
                     "IM": Europe,          # Isle of Man
                     "IL": AsiaTemperate,   # Israel
                     "IT": Europe,          # Italy
                     "JM": Caribbean,       # "Jamaica"
                     "JP": AsiaTemperate,   # Japan
                     "JE": Europe,          # Jersey
                     "JO": AsiaTemperate,   # Jordan
                     "KZ": AsiaTemperate,   # Kazakhstan
                     "KE": Africa,          # Kenya
                     "KI": Pacific,         # Kiribati
                     "KP": AsiaTemperate,   # North Korea
                     "KR": AsiaTemperate,   # South Korea
                     "KW": AsiaTemperate,   # Kuwait
                     "KG": AsiaTemperate,   # Kyrgyzstan
                     "LA": AsiaTropical,    # Laos
                     "LV": Europe,          # Latvia
                     "LB": AsiaTemperate,   # Lebanon
                     "LT": Europe,          # Lithuania
                     "LS": Africa,          # Lesotho
                     "LR": Africa,          # Liberia
                     "LY": Africa,          # Libya
                     "LI": Europe,          # Liechtenstein		LIE	438
                     "LU": Europe,          # Luxembourg		LUX	442
                     "MO": AsiaTemperate,   # Macao		MAC	446
                     "MG": Africa,          # Madagascar		MDG	450
                     "MW": Africa,          # Malawi		MWI	454
                     "MY": AsiaTropical,    # Malaysia		MYS	458
                     "MV": Africa,          # Maldives	MV	MDV	462
                     "ML": Africa,          # Mali	ML	MLI	466
                     "MT": Europe,          # Malta	MT	MLT	470
                     "MH": Pacific,         # Marshall Islands (the)		MHL	584
                     "MQ": Caribbean,       # Martinique		MTQ	474
                     "MR": Africa,          # Mauritania		MRT	478
                     "MU": Africa,          # Mauritius		MUS	480
                     "YT": Africa,          # Mayotte		MYT	175
                     "MX": CentralAmerica,  # Mexico		MEX	484
                     "FM": Pacific,         # Micronesia (Federated States of)		FSM	583
                     "MD": Europe,          # Moldova (the Republic of)		MDA	498
                     "MC": Europe,          # Monaco		MCO	492
                     "MN": AsiaTemperate,   # Mongolia		MNG	496
                     "ME": Europe,          # Montenegro		MNE	499
                     "MS": Caribbean,       # Montserrat		MSR	500
                     "MA": Africa,          # Morocco		MAR	504
                     "MZ": Africa,          # Mozambique		MOZ	508
                     "MM": AsiaTropical,    # Myanmar		MMR	104
                     "NA": Africa,          # Namibia		NAM	516
                     "NR": Pacific,         # Nauru		NRU	520
                     "NP": AsiaTemperate,   # Nepal	NP	NPL	524
                     "NL": Europe,          # Netherlands (the)		NLD	528
                     "NC": Pacific,         # New Caledonia		NCL	540
                     "NZ": Australasia,     # New Zealand		NZL	554
                     "NI": CentralAmerica,  # Nicaragua		NIC	558
                     "NE": Africa,          # Niger (the)		NER	562
                     "NG": Africa,          # Nigeria		NGA	566
                     "NU": Pacific,         # Niue		NIU	570
                     "NF": Australasia,     # Norfolk Island		NFK	574
                     "MP": Pacific,         # Northern Mariana Islands (the)		MNP	580
                     "NO": Europe,          # Norway	NO	NOR	578
                     "OM": ArabPeninsula,   # Oman	OM	OMN	512
                     "PK": AsiaTropical,    # Pakistan		PAK	586
                     "PW": Pacific,         # Palau		PLW	585
                     "PS": ArabPeninsula,   # Palestine, State of		PSE	275
                     "PA": CentralAmerica,  # Panama		PAN	591
                     "PG": AsiaTropical,    # Papua New Guinea		PNG	598
                     "PY": SouthAmerica,    # Paraguay		PRY	600
                     "PE": SouthAmerica,    # Peru	PE	PER	604
                     "PH": AsiaTropical,    # Philippines (the)		PHL	608
                     "PN": Pacific,         # Pitcairn		PCN	612
                     "PL": Europe,          # Poland		POL	616
                     "PT": Europe,          # Portugal	PT	PRT	620
                     "PR": Caribbean,       # Puerto Rico		PRI	630
                     "QA": ArabPeninsula,   # Qatar		QAT	634
                     "MK": Europe,          # Republic of North Macedonia		MKD	807
                     "RO": Europe,          # Romania		ROU	642
                     "RU": Europe,          # Russian Federation (the)		RUS	643
                     "RW": Africa,          # Rwanda		RWA	646
                     "RE": Africa,          # Réunion		REU	638
                     "BL": Caribbean,       # Saint Barthélemy		BLM	652
                     "SH": Antarctic,       # Saint Helena, Ascension and Tristan da Cunha		SHN	654
                     "KN": Caribbean,       # Saint Kitts and Nevis		KNA	659
                     "LC": Caribbean,       # Saint Lucia		LCA	662
                     "MF": Caribbean,       # Saint Martin (French part)		MAF	663
                     "PM": NorthAmerica,    # Saint Pierre and Miquelon		SPM	666
                     "VC": Caribbean,       # Saint Vincent and the Grenadines		VCT	670
                     "WS": Pacific,         # Samoa		WSM	882
                     "SM": Europe,          # San Marino		SMR	674
                     "ST": Africa,          # Sao Tome and Principe		STP	678
                     "SA": ArabPeninsula,   # Saudi Arabia		SAU	682
                     "SN": Africa,          # Senegal		SEN	686
                     "RS": Europe,          # Serbia		SRB	688
                     "SC": Africa,          # Seychelles		SYC	690
                     "SL": Africa,          # Sierra Leone		SLE	694
                     "SG": AsiaTropical,    # Singapore		SGP	702
                     "SX": Caribbean,       # Sint Maarten (Dutch part)		SXM	534
                     "SK": Europe,          # Slovakia		SVK	703
                     "SI": Europe,          # Slovenia		SVN	705
                     "SB": Pacific,         # Solomon Islands		SLB	090
                     "SO": Africa,          # Somalia		SOM	706
                     "ZA": Africa,          # South Africa		ZAF	710
                     "GS": Antarctic,       # South Georgia and the South Sandwich Islands		SGS	239
                     "SS": Africa,          # South Sudan		SSD	728
                     "ES": Europe,          # Spain		ESP	724
                     "LK": AsiaTropical,    # Sri Lanka		LKA	144
                     "SD": Africa,          # Sudan (the)		SDN	729
                     "SR": SouthAmerica,    # Suriname		SUR	740
                     "SJ": Europe,          # Svalbard and Jan Mayen		SJM	744
                     "SE": Europe,          # Sweden		SWE	752
                     "CH": Europe,          # Switzerland	CH	CHE	756
                     "SY": ArabPeninsula,   # Syrian Arab Republic		SYR	760
                     "TW": AsiaTemperate,   # Taiwan (Province of China)		TWN	158
                     "TJ": AsiaTemperate,   # Tajikistan	TJ	TJK	762
                     "TZ": Africa,          # Tanzania, United Republic of	TZ	TZA	834
                     "TH": AsiaTropical,    # Thailand		THA	764
                     "TL": AsiaTropical,    # Timor-Leste		TLS	626
                     "TG": Pacific,         # Togo		TGO	768
                     "TK": Pacific,         # Tokelau		TKL	772
                     "TO": Pacific,         # Tonga		TON	776
                     "TT": Caribbean,       # Trinidad and Tobago		TTO	780
                     "TN": Africa,          # Tunisia		TUN	788
                     "TR": AsiaTemperate,   # Turkey		TUR	792
                     "TM": AsiaTemperate,   # Turkmenistan		TKM	795
                     "TC": Caribbean,       # Tu# rks and Caicos Islands (the)		TCA	796
                     "TV": Pacific,         # T# uvalu		TUV	798
                     "UG": Africa,          # Uganda		UGA	800
                     "UA": Europe,          # Ukraine		UKR	804
                     "AE": ArabPeninsula,   # United Arab Emirates (the)		ARE	784
                     "GB": Europe,          # United Kingdom of Great Britain and Northern Ireland (the)	GB	GBR	826
                     "UM": Pacific,         # United States Minor Outlying Islands (the)		UMI	581
                     "US": NorthAmerica,    # United States of America (the)		USA	840
                     "UY": SouthAmerica,    # Uruguay		URY	858
                     "UZ": AsiaTemperate,   # Uzbekistan		UZB	860
                     "VU": Pacific,         # Vanuatu		VUT	548
                     "VE": SouthAmerica,    # Venezuela (Bolivarian Republic of)
                     "VN": AsiaTropical,    # Viet Nam
                     "VG": Caribbean,       # Virgin Islands (British)
                     "VI": Caribbean,       # Virgin Islands (U.S.)
                     "WF": Pacific,         # Wallis and Futuna
                     "EH": Africa,          # Western Sahara	EH	ESH	732
                     "YE": ArabPeninsula,   # Yemen	YE
                     "ZM": Africa,          # Zambia
                     "ZW": Africa}          # Zimbabwe


def _country_code_to_continent(country):
    global country_code_dict
    if country not in country_code_dict:
        raise RuntimeError(f"Don't know how to assign '{country}' to a continent")
    return country_code_dict[country]


def lat_lng_to_continent(lat, lng):
    import reverse_geocoder as rg
    coords = (lat, lng),
    country = rg.search(coords)[0]['cc']
    return _country_code_to_continent(country)


def _parse_lat_lng_impl(s, pos_char, neg_char, context, minn, maxx):
    # Try direct parsing as float
    res = None
    try:
        res = float(s)
    except ValueError:
        # Also accept "N" and "S" or "E" and "W"
        if s.upper().endswith(pos_char) or s.upper().endswith(neg_char):
            try:
                v = float(s[:-1])
                m = 1.0 if s.upper().endswith(pos_char) else -1.0
                res = v*m
            except ValueError:
                pass
    if res is None or res < minn or res > maxx:
        raise RuntimeError(f"Could not parse {s} as a {context}. It should be a number between {minn} and {maxx}")
    return res


def parse_lat(lat_str):
    return _parse_lat_lng_impl(lat_str, "N", "S", "latitude", -90, 90)


def parse_lng(lng_str):
    return _parse_lat_lng_impl(lng_str, "E", "W", "longitude", -180, 180)
