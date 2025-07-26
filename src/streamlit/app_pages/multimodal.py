import streamlit as st

from assets import style
from assets import PATHS

sections = [
    "Un bout de lecture",
    "Un bout d'art",
    "Quelques données",
    "Et un beau graphique"
    ]

section_counter = 0
def next_section():
    global section_counter
    st.markdown(f'<a name="{sections[section_counter]}"></a>', unsafe_allow_html=True)
    st.markdown(f"<hr style='{style.divider}'>",unsafe_allow_html=True)
    # st.markdown("""<div style="padding-top: 240px; margin-top: -240px;"></div>""",unsafe_allow_html=True)
    st.header(sections[section_counter])
    section_counter += 1


def show():
    st.title("Page de présentation")
    st.subheader("Bienvenue 👋")
    st.write("Ceci est une page d'accueil avec du contenu lorem ipsum.")

    next_section()
    st.markdown(f"""Lorsque j’avais six ans j’ai vu, une fois, une magnifique
image, dans un livre sur la Forêt Vierge qui s’appelait « His
toires Vécues ». Ça représentait un serpent boa qui avalait un
fauve. Voilà la copie du dessin.
On disait dans le livre : « Les serpents boas avalent leur
proie tout entière, sans la mâcher. Ensuite ils ne peuvent plus
bouger et ils dorment pendant les six mois de leur digestion. »
J’ai alors beaucoup réfléchi sur les aventures de la jungle
et, à mon tour, j’ai réussi, avec un crayon de couleur, à tracer
mon premier dessin. Mon dessin numéro 1. Il était comme ça :
J’ai montré mon chef-d’œuvre aux grandes personnes et je
leur ai demandé si mon dessin leur faisait peur.
Elles m’ont répondu : « Pourquoi un chapeau ferait-il
peur ? »
{style.highlight('Mon dessin ne représentait pas un chapeau.')} Il représentait
un serpent boa qui digérait un éléphant. J’ai alors dessiné
l’intérieur du serpent boa, afin que les grandes personnes puis
sent comprendre. Elles ont toujours besoin d’explications. Mon
dessin numéro 2 était comme ça :
Les grandes personnes m’ont conseillé de laisser de côté les
dessins de serpents boas ouverts ou fermés, et de m’intéresser
plutôt à la géographie, à l’histoire, au calcul et à la grammaire.
C’est ainsi que j’ai abandonné, à l’âge de six ans, une magnifique
carrière de peintre. J’avais été découragé par l’insuccès de mon
dessin numéro 1 et de mon dessin numéro 2. Les grandes per-
sonnes ne comprennent jamais rien toutes seules, et c’est fati
gant, pour les enfants, de toujours et toujours leur donner des
explications.
J’ai donc dû choisir un autre métier et j’ai appris à piloter
des avions. J’ai volé un peu partout dans le monde. Et la géo-
graphie, c’est exact, m’a beaucoup servi. Je savais reconnaître,
du premier coup d’œil, la Chine de l’Arizona. C’est très utile, si
l’on est égaré pendant la nuit.
J’ai ainsi eu, au cours de ma vie, des tas de contacts avec
des tas de gens sérieux. J’ai beaucoup vécu chez les grandes per
sonnes. Je les ai vues de très près. Ça n’a pas trop amélioré mon
opinion.
Quand j’en rencontrais une qui me paraissait un peu lu-
cide, je faisais l’expérience sur elle de mon dessin numéro 1 que
j’ai toujours conservé. Je voulais savoir si elle était vraiment
compréhensive. Mais toujours elle me répondait : « C’est un
chapeau. » Alors je ne lui parlais ni de serpents boas, ni de fo-
rêts vierges, ni d’étoiles. Je me mettais à sa portée. Je lui parlais
de bridge, de golf, de politique et de cravates. Et la grande per-
sonne était bien contente de connaître un homme aussi raison-
nable.""", unsafe_allow_html=True)
    next_section()
    st.image(PATHS.streamlit_images / "joconde.jpg", caption="La Joconde")
    st.text("""La Joconde est le portrait le plus célèbre au monde. L'identité du modèle est régulièrement remise en question, mais on admet généralement qu'il s'agit d'une dame florentine, prénommée Lisa, épouse de Francesco del Giocondo. Le nom Giocondo a été très tôt francisé en Joconde, mais le tableau est aussi connu sous le titre de Portrait de Monna Lisa, monna signifiant dame ou madame en italien ancien.

UNE LONGUE HISTOIRE AVEC LA FRANCE
C'est probablement entre 1503 et 1506 que Francesco del Giocondo commande le portrait de sa jeune épouse à Léonard qui réside alors à Florence. Mais il ne l'a certainement jamais eu en sa possession. En effet, Léonard, invité à la cour de François Ier en 1517, l'emporte sans doute avec lui en France où il meurt deux ans plus tard au Clos Lucé, à Amboise. Le tableau est vraisemblablement acheté par François Ier lui-même, qui admire « le sourire quasi divin » de la dame. Il devient rapidement par la suite une œuvre emblématique des collections françaises.

UNE COMPOSITION CLAIRE
Le tableau représente la jeune femme de trois quarts, assise dans une loggia ouverte sur un paysage. Elle regarde le spectateur et sourit. L'avant-bras gauche appuyé sur l'accoudoir d'un fauteuil, les mains posées l'une sur l'autre, elle domine l'ensemble de la composition. Sa silhouette s'inscrit dans une forme pyramidale qui affermit la stabilité de la figure. Les cheveux sombres, recouverts d'un léger voile, encadrent le visage aux sourcils épilés qui attire toute l'attention du spectateur.

UNE PRÉSENCE « QUI CRÈVE L'ÉCRAN »
Avec son regard pénétrant et son léger sourire, Monna Lisa semble défier le spectateur et s'en amuser. Léonard a su capter une expression fugace passée sur le visage de la jeune femme. Il représente avec précision les muscles de son visage et tous leurs mouvements, notamment aux contours des yeux et aux commissures des lèvres. Son habileté réside surtout dans la manière dont il travaille le volume des carnations, en estompant de manière très subtile les passages de l'ombre à la lumière.
Il invente ainsi un nouvel effet, le sfumato, qui lui permet de mieux inscrire la figure dans l'espace. C'est principalement grâce à cet effet, caractéristique de la peinture de Léonard, que la Joconde apparaît si présente au spectateur. Elle est là toute proche et nous observe comme derrière une fenêtre. Cette présence est encore accentuée par le contraste fort qui existe à l'intérieur du tableau entre la figure et le paysage vaporeux sur lequel sa silhouette se détache.

UN PAYSAGE ÉNIGMATIQUE
Le vaste paysage montre de lointaines vallées et des pitons rocheux perdus dans la brume. Sa profondeur est obtenue grâce à une perspective atmosphérique qui consiste à créer différents plans en modulant progressivement les tonalités. On passe ainsi d'un brun verdâtre à un vert bleuté pour finalement rejoindre le ciel. Au plan le plus rapproché, des signes de civilisation apparaissent : sur la droite, un pont enjambe une rivière , sur la gauche, un sentier serpente. Mais au fur et à mesure que l'on se rapproche de la ligne d'horizon, des montagnes grandioses apparaissent, puis se fondent dans une lumière vaporeuse et vibrante.

UNE ÉTERNELLE FASCINATION
Monna Lisa nous observe et nous sourit, mais son regard s'efface derrière l'icône qu'elle est devenue. Elle fascine. Chacun y projette ses propres fantasmes. Les artistes, de toutes les périodes, n'ont cessé de s'en inspirer, de Raphaël à Corot, de Marcel Duchamp à Jean-Michel Basquiat. Qu'elle soit référence absolue ou objet de raillerie, elle reste à jamais un phare dans l'histoire de l'art."""
)
    # display image
    next_section()

    import pandas as pd
    from sklearn.datasets import load_iris
    iris = load_iris(as_frame=True)
    df = iris.frame
    st.subheader("🌸 Base de données Iris")

    st.text("Code utilisé pour télécharger:")
    st.code("""iris = load_iris(as_frame=True)
    df = iris.frame""")

    # Aperçu du dataframe
    st.markdown("#### Aperçu des données")
    st.dataframe(df.head(10))

    # disply dataframe

    next_section()
    # display graph
    from matplotlib import pyplot as plt
    plt.style.use('dark_background')

    df["target_name"] = df["target"].map(dict(enumerate(iris.target_names)))

    species = iris.target_names
    color_map = {name: style.graph_colors[i] for i, name in enumerate(species)}
    fig, ax = plt.subplots(figsize=(8, 6), facecolor="none")

    for name in species:
        sub_df = df[df["target_name"] == name]
        ax.scatter(
            sub_df["petal length (cm)"],
            sub_df["petal width (cm)"],
            label=name,
            color=color_map[name],
            alpha=0.8,
            edgecolors="k",
            s=100
        )

    # 🎯 Mise en forme
    ax.set_title("Iris - Pétale : Longueur vs Largeur", fontsize=16, fontweight="bold")
    ax.set_xlabel("Longueur du pétale (cm)", fontsize=12)
    ax.set_ylabel("Largeur du pétale (cm)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(title="Espèce")

    # Affichage dans Streamlit
    st.pyplot(fig)

