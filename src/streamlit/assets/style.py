from types import SimpleNamespace
from matplotlib.colors import ListedColormap


#####################
# COULEURS DU THEME #
#####################
# Ces mêmes couleurs ont été utilisées pour remplir le fichier config.toml
# Mais la création de certains styles personnalisés nécessitera d'y faire appel

colors = SimpleNamespace(
    black1 = "#111111",       # background page principale
    grey1 = "#31333E",       # background bandeau navigation+background menus déroulants
    grey2 = "#818494",       # texte adouci(titre bandeau nav)+icone suppression non actif dans multiselects
    grey3 = "#A4A8B7",       # 
    white1 = "#FAFAFA",      # corps de texte
    yellow1 = "#ECB959",      # texte mis en évidence
    blue1 = "#56BDED",       # lignes de séparation entre les parties
    blue2 = "#5098F8",       # texte emphase légère
    pink1 = "#E24768"       # surbrillance en sélection
)

# fonction à utiliser pour surligner le texte
def highlight(text, color=colors.yellow1, bold=False, italic=False):
    style = f"color: {color};"
    if bold:
        style += " font-weight: bold;"
    if italic:
        style += " font-style: italic;"
    return f"<span style='{style}'>{text}</span>"

#########################
# COLORMAP PERSONALISEE #
#########################

graph_colors = [
    "#93C7FA",
    "#2B66C2",
    "#F3AFAD",
    "#EA4339",
    "#9AECA8",
    "#57AD9D",
    "#F8D37A"
]
colormap = ListedColormap(graph_colors, name='my_palette')

##############
# STYLES CSS #
##############
divider = f"""
    border: none;
    border-top: 3px solid {colors.blue1};
"""

navbar_link = f"""
            text-decoration: none;
            display: block;
            padding: 0.5rem 1rem;
            margin: 0.3rem 0;
            border-radius: 5px;
            font-weight: 500;
            color: {colors.white1};
        """

