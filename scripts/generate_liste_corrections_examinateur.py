"""Generate the Word file requested by the examiner without external packages."""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile


OUTPUT = Path("rapport/Liste_complete_corrections_apportees_au_memoire.docx")
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def run(text: str, *, bold: bool = False, italic: bool = False, size: int = 22, color: str | None = None) -> str:
    properties = [f'<w:sz w:val="{size}"/>', f'<w:szCs w:val="{size}"/>']
    if bold:
        properties.append("<w:b/>")
    if italic:
        properties.append("<w:i/>")
    if color:
        properties.append(f'<w:color w:val="{color}"/>')
    return f'<w:r><w:rPr>{"".join(properties)}</w:rPr><w:t xml:space="preserve">{escape(text)}</w:t></w:r>'


def paragraph(
    text: str = "",
    *,
    bold: bool = False,
    italic: bool = False,
    size: int = 22,
    align: str | None = None,
    before: int = 0,
    after: int = 110,
    color: str | None = None,
) -> str:
    props = []
    if align:
        props.append(f'<w:jc w:val="{align}"/>')
    props.append(f'<w:spacing w:before="{before}" w:after="{after}"/>')
    return f'<w:p><w:pPr>{"".join(props)}</w:pPr>{run(text, bold=bold, italic=italic, size=size, color=color)}</w:p>'


def cell(text: str, width: int, *, header: bool = False) -> str:
    shade = '<w:shd w:val="clear" w:fill="1F4E78"/>' if header else ''
    cell_props = f'<w:tcPr><w:tcW w:w="{width}" w:type="dxa"/>{shade}<w:vAlign w:val="top"/></w:tcPr>'
    paragraphs = []
    for line in text.split("\n"):
        paragraphs.append(
            paragraph(
                line,
                bold=header,
                size=18 if not header else 18,
                color="FFFFFF" if header else None,
                after=0,
            )
        )
    return f'<w:tc>{cell_props}{"".join(paragraphs)}</w:tc>'


def table(headers: list[str], rows: list[tuple[str, str, str, str]]) -> str:
    widths = [450, 2280, 4620, 2370]
    grid = ''.join(f'<w:gridCol w:w="{width}"/>' for width in widths)
    border = (
        '<w:tblBorders>'
        '<w:top w:val="single" w:sz="6" w:color="B7C9D6"/>'
        '<w:left w:val="single" w:sz="6" w:color="B7C9D6"/>'
        '<w:bottom w:val="single" w:sz="6" w:color="B7C9D6"/>'
        '<w:right w:val="single" w:sz="6" w:color="B7C9D6"/>'
        '<w:insideH w:val="single" w:sz="4" w:color="D9E2F3"/>'
        '<w:insideV w:val="single" w:sz="4" w:color="D9E2F3"/>'
        '</w:tblBorders>'
    )
    props = (
        '<w:tblPr><w:tblW w:w="9720" w:type="dxa"/>'
        '<w:tblLayout w:type="fixed"/>'
        f'{border}'
        '<w:tblCellMar><w:top w:w="70" w:type="dxa"/><w:left w:w="70" w:type="dxa"/>'
        '<w:bottom w:w="70" w:type="dxa"/><w:right w:w="70" w:type="dxa"/></w:tblCellMar>'
        '</w:tblPr>'
    )
    header_row = '<w:tr>' + ''.join(cell(text, width, header=True) for text, width in zip(headers, widths)) + '</w:tr>'
    body_rows = []
    for row in rows:
        body_rows.append('<w:tr>' + ''.join(cell(text, width) for text, width in zip(row, widths)) + '</w:tr>')
    return f'<w:tbl>{props}<w:tblGrid>{grid}</w:tblGrid>{header_row}{"".join(body_rows)}</w:tbl>'


def document_xml() -> str:
    rows = [
        (
            "1",
            "Cadrage du thème et du périmètre",
            "Le mémoire est recentré sur la conception et l'opérationnalisation d'un pipeline hybride de recommandation emploi-compétences. Il est présenté comme une démonstration technique appliquée à KmerAI, et non comme un diagnostic général du marché du travail camerounais ni comme une preuve de réussite de recrutement.",
            "Avant-propos, résumé et Introduction générale : contexte, problématique, objectifs, méthodologie et conclusion.",
        ),
        (
            "2",
            "Contexte camerounais rendu chiffré",
            "Insertion de données EESI3 sourcées : taux d'emploi de 50,8 %, chômage BIT de 6,1 %, sous-utilisation SU2 de 23,0 % et SU4 de 26,3 %, situation plus défavorable des jeunes de 15 à 34 ans et part de l'emploi informel de 86,6 %. Les indicateurs sont définis comme complémentaires et non additionnables.",
            "Introduction générale, section « Contexte et justification », pp. 1--2 de la version révisée.",
        ),
        (
            "3",
            "Intégration du FNE et de l'OIT",
            "Ajout du rôle du Fonds National de l'Emploi dans l'intermédiation, l'orientation et l'insertion. Le bilan 2023 (53 375 prospections et 21 719 insertions) est présenté comme un flux administratif, non comme une création nette d'emplois ou une mesure de durabilité des placements. Le Programme Pays de Travail Décent de l'OIT est également mobilisé.",
            "Introduction générale, section « Contexte et justification », p. 2 ; bibliographie.",
        ),
        (
            "4",
            "Problématique, questions et objectifs précisés",
            "La question centrale et les objectifs distinguent explicitement le rappel sémantique, la vérification relationnelle par graphe, la génération contrôlée et la décision humaine. Les objectifs sont formulés sans codes abrégés et sans promettre une automatisation du recrutement ou de l'orientation.",
            "Introduction générale : « Problématique », « Objectifs spécifiques », « Intérêt de l'étude » et « Méthodologie de l'étude ».",
        ),
        (
            "5",
            "Rôles des composantes techniques séparés",
            "pgvector est présenté comme mécanisme de rappel sémantique, Neo4j comme support de contrôle relationnel, de calcul du skill gap et de reranking, et le LLM comme composante de routage et de formulation à partir de preuves. Le LLM ne décide pas seul de la pertinence d'une recommandation.",
            "Méthodologie et implémentation : workflow hybride, GraphRAG, critic et génération contrôlée.",
        ),
        (
            "6",
            "Provenance et limites des données clarifiées",
            "La formulation sur les offres a été corrigée : il s'agit d'un corpus du projet constitué à partir de plateformes d'emploi, dont celle du FNE, sans suggérer un partenariat ou une collecte directe non démontrée. Les données manquantes ne sont pas inventées et les limites de descriptions sont explicitées.",
            "Chapitre méthodologique : « Sources de données et leur rôle dans le système » et préparation des données.",
        ),
        (
            "7",
            "Fine-tuning et préparation documentaire mieux justifiés",
            "Les contraintes de calcul, le choix du meilleur checkpoint, l'écart entre apprentissage et test, ainsi que la nature interprétable des chunks documentaires sont explicités. Les figures correspondantes ont été réorganisées ou redimensionnées pour améliorer leur lisibilité, sans modifier les résultats.",
            "Chapitre 3 : fine-tuning SentenceTransformer et figures 3.3, 3.6 et 3.7.",
        ),
        (
            "8",
            "Interprétation des résultats et des métriques rendue prudente",
            "Les valeurs de Recall, NDCG, Precision et MRR sont présentées comme des mesures internes fondées sur proxy_relevance. Elles ne sont ni assimilées à un taux de recrutement réussi ni à une validation par des recruteurs. L'ablation précise que Neo4j réordonne le même vivier présélectionné par pgvector.",
            "Chapitre 4 : sections 4.2 et 4.4.1 ; conclusion générale.",
        ),
        (
            "9",
            "Calibration Optuna et critic encadrés",
            "Les pondérations optimisées sont présentées comme dépendantes de l'objectif, du proxy et de l'échantillon. Le critic vérifie l'ancrage lexical dans le contexte récupéré, non la vérité métier. Le besoin d'une validation indépendante, hors échantillon et avec annotations professionnelles est explicite.",
            "Chapitre 4 : sections 4.4.2 et 4.5 ; limites et perspectives.",
        ),
        (
            "10",
            "Positionnement rigoureux des GNN",
            "Le mémoire n'affirme pas entraîner ni évaluer une GNN. Une GNN hétérogène de type relationnel est introduite comme perspective conditionnée à l'existence d'interactions horodatées et validées, à une comparaison temporellement séparée et à des annotations professionnelles. Les risques liés aux noeuds très connectés sont signalés.",
            "Chapitre 1, sous-section « Graphes de connaissances et recommandation relationnelle ».",
        ),
        (
            "11",
            "Tableaux et figures rendus plus lisibles",
            "Les statistiques du contexte ont été conservées sous forme de paragraphes pour préserver la fluidité de l'introduction. La figure 4.5 présente désormais ses deux graphiques côte à côte sur une même ligne. D'autres figures techniques ont été redimensionnées ou replacées au plus près de leur explication.",
            "Introduction générale, pp. 1--2 ; chapitres 1 et 3 ; figure 4.5, p. 59.",
        ),
        (
            "12",
            "Références bibliographiques complétées",
            "Ajout et citation de sources primaires ou institutionnelles : INS/EESI3, FNE, rapport gouvernemental sur les insertions, OIT/PPTD, OIT jobs gap, R-GCN et JobXMLC. Les sources permettent de vérifier les chiffres, les limites et la perspective GNN.",
            "Bibliographie et citations intégrées dans l'introduction et la revue de littérature.",
        ),
        (
            "13",
            "Corrections de forme et contrôle de compilation",
            "Des corrections ciblées de syntaxe LaTeX, d'espacement et de typographie ont été appliquées, puis le PDF a été reconstruit. Lors du dernier contrôle, la compilation ne présentait ni erreur bloquante ni citation ou référence non résolue.",
            "Ensemble du mémoire ; version PDF révisée jointe séparément.",
        ),
    ]
    body = [
        paragraph("RELEVÉ COMPLET DES CORRECTIONS INTÉGRÉES DANS LA VERSION RÉVISÉE", bold=True, size=30, align="center", after=120, color="1F4E78"),
        paragraph("Version révisée destinée à être transmise à l'attention du Pr MELATAGIA Paulin", bold=True, size=22, align="center", after=240),
        paragraph("Mémoire professionnel : Conception d'un pipeline hybride de recherche sémantique pour les recommandations d'emploi-compétences, fondé sur les LLMs et les graphes de connaissances", italic=True, size=20, align="center", after=250),
        paragraph("Auteur : NGOULOU NGOUBILI Irch Defluviaire", size=20, after=30),
        paragraph("Date : 6 août 2026", size=20, after=230),
        paragraph("Objet du document", bold=True, size=24, color="1F4E78", before=80, after=80),
        paragraph("Le présent document récapitule les corrections de fond, de méthode, de forme et de traçabilité bibliographique effectivement intégrées dans la version révisée du mémoire. Il est organisé par point révisé et indique les emplacements permettant leur vérification directe.", size=21, after=180),
        table(["N°", "Point révisé", "Correction intégrée", "Emplacement dans le mémoire"], rows),
        paragraph("Précision méthodologique", bold=True, size=24, color="1F4E78", before=220, after=80),
        paragraph("Les résultats quantitatifs conservés dans le mémoire décrivent l'évaluation interne du prototype. Ils ne sont pas présentés comme une validation de recrutement, ni comme une preuve d'efficacité généralisable sans jeu indépendant et annotation métier.", size=21, after=80),
        paragraph("Le document a été recompilé après intégration des corrections ; les citations et références ont été vérifiées.", size=21, italic=True, after=0),
        '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134" w:header="708" w:footer="708" w:gutter="0"/></w:sectPr>',
    ]
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:document xmlns:w="{W_NS}"><w:body>{"".join(body)}</w:body></w:document>'
    )


def write_docx(output: Path) -> None:
    content_types = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>'''
    package_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>'''
    core = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>Liste complète des corrections apportées au mémoire</dc:title>
  <dc:creator>NGOULOU NGOUBILI Irch Defluviaire</dc:creator>
  <cp:lastModifiedBy>NGOULOU NGOUBILI Irch Defluviaire</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">2026-08-06T00:00:00Z</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">2026-08-06T00:00:00Z</dcterms:modified>
</cp:coreProperties>'''
    app = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"><Application>Microsoft Office Word</Application></Properties>'''
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", package_rels)
        archive.writestr("docProps/core.xml", core)
        archive.writestr("docProps/app.xml", app)
        archive.writestr("word/document.xml", document_xml())


if __name__ == "__main__":
    write_docx(OUTPUT)
    print(OUTPUT.resolve())
