from __future__ import annotations

import re
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_XLSX = ROOT / "data" / "raw" / "vrai_data.xlsx"
OUT = ROOT / "outputs" / "memoire_stats" / "vrai_data_offres"
FIG = ROOT / "rapport" / "figures" / "generated" / "vrai_data_offres"
PROC = ROOT / "data" / "processed"


CAMEROON_CITIES = {
    "abong mbang": "Abong-Mbang",
    "bafang": "Bafang",
    "bafia": "Bafia",
    "bafoussam": "Bafoussam",
    "bamenda": "Bamenda",
    "bangangte": "Bangangté",
    "bertoua": "Bertoua",
    "buea": "Buéa",
    "buéa": "Buéa",
    "douala": "Douala",
    "dschang": "Dschang",
    "ebolowa": "Ebolowa",
    "edea": "Édéa",
    "edéa": "Édéa",
    "ekoumdoum": "Yaoundé (Ekoumdoum)",
    "foumban": "Foumban",
    "garoua": "Garoua",
    "kribi": "Kribi",
    "kumbo": "Kumbo",
    "limbe": "Limbé",
    "limbé": "Limbé",
    "maroua": "Maroua",
    "mbalmayo": "Mbalmayo",
    "ngaoundere": "Ngaoundéré",
    "ngaoundéré": "Ngaoundéré",
    "ngousso": "Ngousso",
    "nkongsamba": "Nkongsamba",
    "yaound": "Yaoundé",
    "yaounde": "Yaoundé",
    "yaoundé": "Yaoundé",
}

INTERNATIONAL_CITIES = {
    "abidjan": "Abidjan",
    "bouenza": "Bouenza",
    "brazzaville": "Brazzaville",
    "kinshasa": "Kinshasa",
    "libreville": "Libreville",
    "lome": "Lomé",
    "lomé": "Lomé",
    "pointe noire": "Pointe-Noire",
    "pointe-noire": "Pointe-Noire",
}

NO_CITY_MARKERS = {
    "",
    "nan",
    "none",
    "non renseigne",
    "non renseigné",
    "cameroun",
    "cameroun ville non precisee",
    "cameroun ville non précisée",
    "ville non precisee",
    "ville non précisée",
}


def strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def clean_text(value) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def canonical(value) -> str:
    text = strip_accents(clean_text(value)).casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def title_preserve_acronyms(value: str) -> str:
    text = clean_text(value)
    if not text:
        return ""
    words = []
    for word in re.split(r"(\s+|-)", text.lower()):
        if word.isspace() or word == "-":
            words.append(word)
        elif len(word) <= 4 and word.upper() in {"FNE", "ONG", "RH", "TIC", "IT", "QHSE", "HSE", "BTP"}:
            words.append(word.upper())
        else:
            words.append(word[:1].upper() + word[1:])
    return "".join(words)


def normalize_source(value) -> str:
    text = clean_text(value)
    key = canonical(text)
    if not key:
        return "Non renseigné"
    if "fne" in key:
        return "FNE"
    if "linkedin" in key:
        return "LinkedIn"
    if "minajobs" in key:
        return "MinaJobs"
    if "emploi" in key and "cm" in key:
        return "Emploi.cm"
    return title_preserve_acronyms(text)


def normalize_country(value) -> str:
    key = canonical(value)
    if not key:
        return "Non renseigné"
    mapping = {
        "cameroun": "Cameroun",
        "cameroon": "Cameroun",
        "congo": "Congo",
        "congo brazzaville": "Congo",
        "republique du congo": "Congo",
        "rdc": "RDC",
        "republique democratique du congo": "RDC",
        "etats unis": "États-Unis",
        "etats unis d amerique": "États-Unis",
        "gabon": "Gabon",
        "cote d ivoire": "Côte d'Ivoire",
        "benin": "Bénin",
        "togo": "Togo",
    }
    return mapping.get(key, title_preserve_acronyms(clean_text(value)))


def split_multivalue(value) -> list[str]:
    text = clean_text(value)
    if not text:
        return []
    text = re.sub(r"\s+(?:et|ou)\s+", ",", text, flags=re.IGNORECASE)
    parts = re.split(r"[,;|/\n]+", text)
    return [clean_text(part) for part in parts if clean_text(part)]


def normalize_city_one(value, country: str) -> str:
    key = canonical(value)
    if key in {"international", "a l etranger"}:
        return "International non localisée"
    if key in NO_CITY_MARKERS:
        return "Non précisée"
    if key in CAMEROON_CITIES:
        return CAMEROON_CITIES[key]
    if key in INTERNATIONAL_CITIES:
        return INTERNATIONAL_CITIES[key]
    if country == "Cameroun":
        return title_preserve_acronyms(clean_text(value))
    return title_preserve_acronyms(clean_text(value))


def normalize_city_list(value, country: str) -> list[str]:
    cities = []
    seen = set()
    for part in split_multivalue(value):
        city = normalize_city_one(part, country)
        key = canonical(city)
        if city and key not in seen:
            seen.add(key)
            cities.append(city)
    real_cities = [city for city in cities if city != "Non précisée"]
    if real_cities:
        cities = real_cities
    if not cities:
        cities = [f"{country} (ville non précisée)" if country else "Non précisée"]
    if cities == ["Non précisée"] and country:
        cities = [f"{country} (ville non précisée)"]
    return cities


def normalize_skill(value: str) -> str:
    text = clean_text(value)
    text = text.strip("[](){}")
    text = re.sub(r"^[\"']|[\"']$", "", text)
    key = canonical(text)
    if not key or key in {"nan", "none"}:
        return ""
    if key.startswith("doit dispenser") or key.startswith("doit avoir"):
        return ""
    if len(text) > 80 and len(text.split()) > 9:
        return ""
    aliases = {
        "finance": "Finance",
        "administration": "Administration",
        "informatique": "Informatique",
        "comptabilite": "Comptabilité",
        "tic": "TIC",
        "rh": "Ressources humaines",
        "ong": "ONG",
    }
    return aliases.get(key, title_preserve_acronyms(text))


def normalize_skills(value) -> list[str]:
    text = clean_text(value)
    if not text:
        return []
    quoted = re.findall(r"'([^']+)'|\"([^\"]+)\"", text)
    if quoted:
        raw_parts = [a or b for a, b in quoted]
    else:
        raw_parts = split_multivalue(text)
    skills = []
    seen = set()
    for part in raw_parts:
        skill = normalize_skill(part)
        key = canonical(skill)
        if skill and key not in seen:
            seen.add(key)
            skills.append(skill)
    return skills


def pct(series: pd.Series) -> pd.Series:
    return (series / series.sum() * 100).round(2)


def save_table(df: pd.DataFrame, name: str) -> None:
    df.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")


def setup_plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 230,
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "legend.frameon": False,
            "axes.titleweight": "bold",
        }
    )
    fmt = FuncFormatter(lambda x, _pos=None: f"{int(x):,}".replace(",", " "))
    return plt, fmt


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    PROC.mkdir(parents=True, exist_ok=True)
    plt, fmt = setup_plot()
    colors = {
        "blue": "#2457A6",
        "teal": "#008B8B",
        "orange": "#E68619",
        "green": "#3A7D44",
        "red": "#B13E3E",
        "gray": "#5F6B7A",
    }

    df = pd.read_excel(RAW_XLSX, sheet_name="Offres_FNE", dtype=str)
    expected = [
        "source_raw",
        "lien_reference",
        "titre_poste",
        "employeur",
        "type_entreprise",
        "pays_raw",
        "ville_region_raw",
        "secteur_raw",
        "groupe_contrat",
        "type_contrat",
        "niveau_experience",
        "niveau_etudes",
        "competences_raw",
        "details_raw",
    ]
    if df.shape[1] >= len(expected):
        df = df.rename(columns=dict(zip(df.columns[: len(expected)], expected)))

    for col in df.columns:
        df[col] = df[col].map(clean_text)
        df[col] = df[col].replace("", pd.NA)

    df.insert(0, "source_row_number", range(1, len(df) + 1))
    df["source_clean"] = df["source_raw"].map(normalize_source)
    df["pays_clean"] = df["pays_raw"].map(normalize_country)
    df["villes_list_clean"] = df.apply(
        lambda r: normalize_city_list(r.get("ville_region_raw"), r.get("pays_clean")),
        axis=1,
    )
    df["ville_principale_clean"] = df["villes_list_clean"].str[0]
    df["zone_localisation"] = df.apply(
        lambda r: "International"
        if r["ville_principale_clean"] == "International non localisée"
        else "Non précisée"
        if r["ville_principale_clean"] == "Non précisée"
        else ("Cameroun" if r["pays_clean"] == "Cameroun" else "International"),
        axis=1,
    )
    df["competences_list_clean"] = df["competences_raw"].map(normalize_skills)
    df["details_clean"] = df["details_raw"].fillna("").map(clean_text)
    df["description_disponible"] = df["details_clean"].str.len().ge(40)
    df["titre_clean"] = df["titre_poste"].fillna("").map(title_preserve_acronyms)
    df["secteur_clean"] = df["secteur_raw"].fillna("Non renseigné").map(title_preserve_acronyms)

    df.to_parquet(PROC / "vrai_data_offres_cleaned.parquet", index=False)
    try:
        df.to_excel(PROC / "vrai_data_offres_cleaned.xlsx", index=False)
    except Exception:
        pass

    source = df["source_clean"].value_counts().rename_axis("source").reset_index(name="n")
    source["part_pct"] = pct(source["n"])
    save_table(source, "01_offres_par_source")

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    plot = source.head(12).sort_values("n")
    ax.barh(plot["source"], plot["n"], color=colors["blue"])
    ax.xaxis.set_major_formatter(fmt)
    ax.set_title("Offres par source après nettoyage")
    ax.set_xlabel("Nombre d'offres")
    for y, row in enumerate(plot.itertuples()):
        ax.text(row.n, y, f" {row.part_pct:.1f}%", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "01_offres_par_source.png", bbox_inches="tight")
    plt.close(fig)

    loc = (
        df.groupby(["zone_localisation", "pays_clean", "ville_principale_clean"])
        .size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
    )
    loc["part_pct"] = (loc["n"] / len(df) * 100).round(2)
    save_table(loc, "02_localisation_villes")

    fig, ax = plt.subplots(figsize=(10.8, 6.0))
    top_loc = loc.head(18).sort_values("n")
    color_map = {"Cameroun": colors["green"], "International": colors["orange"], "Non précisée": colors["gray"]}
    ax.barh(
        top_loc["ville_principale_clean"] + " - " + top_loc["pays_clean"],
        top_loc["n"],
        color=top_loc["zone_localisation"].map(color_map),
    )
    ax.xaxis.set_major_formatter(fmt)
    ax.set_title("Localisation nettoyée des offres")
    ax.set_xlabel("Nombre d'offres")
    fig.tight_layout()
    fig.savefig(FIG / "02_localisation_villes.png", bbox_inches="tight")
    plt.close(fig)

    zone = df["zone_localisation"].value_counts().rename_axis("zone").reset_index(name="n")
    zone["part_pct"] = (zone["n"] / len(df) * 100).round(2)
    save_table(zone, "02b_zones_localisation")

    skills = Counter()
    for values in df["competences_list_clean"]:
        skills.update(values)
    skills_df = pd.DataFrame(skills.most_common(25), columns=["competence", "n"])
    skills_df["part_offres_pct"] = (skills_df["n"] / len(df) * 100).round(2)
    save_table(skills_df, "03_competences")

    fig, ax = plt.subplots(figsize=(10.5, 7.0))
    top_skills = skills_df.head(18).sort_values("n")
    ax.barh(top_skills["competence"], top_skills["n"], color=colors["teal"])
    ax.xaxis.set_major_formatter(fmt)
    ax.set_title("Compétences et signaux métiers les plus fréquents")
    ax.set_xlabel("Nombre d'offres")
    fig.tight_layout()
    fig.savefig(FIG / "03_competences.png", bbox_inches="tight")
    plt.close(fig)

    desc_by_source = (
        df.groupby("source_clean")["description_disponible"]
        .agg(n_offres="size", n_description="sum")
        .reset_index()
    )
    desc_by_source["taux_description_pct"] = (
        desc_by_source["n_description"] / desc_by_source["n_offres"] * 100
    ).round(2)
    desc_by_source = desc_by_source.sort_values("n_offres", ascending=False)
    save_table(desc_by_source, "04_description_par_source")

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    plot = desc_by_source.head(12).sort_values("n_offres")
    ax.barh(plot["source_clean"], plot["n_offres"], color=colors["gray"], alpha=0.35, label="Offres")
    ax.barh(plot["source_clean"], plot["n_description"], color=colors["blue"], label="Description disponible")
    ax.xaxis.set_major_formatter(fmt)
    ax.set_title("Disponibilité des descriptions par source")
    ax.set_xlabel("Nombre d'offres")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "04_description_par_source.png", bbox_inches="tight")
    plt.close(fig)

    quality_cols = [
        ("source_raw", "Source"),
        ("titre_poste", "Titre"),
        ("employeur", "Employeur"),
        ("pays_raw", "Pays"),
        ("ville_region_raw", "Ville / région"),
        ("secteur_raw", "Secteur"),
        ("competences_raw", "Compétences"),
        ("details_raw", "Description"),
    ]
    quality = []
    for col, label in quality_cols:
        n = int(df[col].notna().sum())
        quality.append({"champ": label, "n_renseignes": n, "total": len(df), "taux_pct": round(n / len(df) * 100, 2)})
    quality_df = pd.DataFrame(quality)
    save_table(quality_df, "05_qualite_champs_vrai_data")

    summary = pd.DataFrame(
        [
            {"indicateur": "offres", "valeur": len(df)},
            {"indicateur": "sources_distinctes", "valeur": df["source_clean"].nunique()},
            {"indicateur": "pays_distincts", "valeur": df["pays_clean"].nunique()},
            {"indicateur": "villes_distinctes", "valeur": df["ville_principale_clean"].nunique()},
            {"indicateur": "competences_distinctes", "valeur": len(skills)},
            {"indicateur": "descriptions_disponibles", "valeur": int(df["description_disponible"].sum())},
        ]
    )
    save_table(summary, "00_resume_vrai_data")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
