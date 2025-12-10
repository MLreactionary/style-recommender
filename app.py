# app.py
import io
from typing import List, Tuple

from PIL import Image
import streamlit as st

from core import (
    load_trained_models_for_inference,
    predict_pair,
    sample_test_images,
    rank_candidates_for_anchor,
    score_outfit,
    IMAGES_DIR,
    call_ollama,
    analyze_style,
    extract_color_palette,
    get_test_image_pool,
)


st.set_page_config(page_title="Fashion CLIP Stylist", layout="wide")
st.title("Stylo AI")

if "pair_results" not in st.session_state:
    st.session_state["pair_results"] = None

if "matches_ranked" not in st.session_state:
    st.session_state["matches_ranked"] = None
if "matches_anchor_desc" not in st.session_state:
    st.session_state["matches_anchor_desc"] = ""

if "outfit_results" not in st.session_state:
    st.session_state["outfit_results"] = None

if "style_results" not in st.session_state:
    st.session_state["style_results"] = None

if "outfit_items" not in st.session_state:
    st.session_state["outfit_items"] = []



@st.cache_resource
def _load_models_cached():
    return load_trained_models_for_inference()


clip_model, preprocess, classifier = _load_models_cached()


def _load_dataset_image(rel_path: str) -> Image.Image:
    return Image.open(IMAGES_DIR / rel_path).convert("RGB")


def _llm_stylist_for_pair(
    desc1: str, desc2: str, cos_sim: float, prob: float
) -> str:
    from math import isnan

    if isnan(prob):
        prob_val = "N/A"
    else:
        prob_val = f"{prob*100:.1f}%"

    prompt = f"""
You are a friendly fashion stylist.

Two clothing items are being evaluated by an ML model trained on Polyvore outfits.
The model outputs:
- CLIP cosine similarity: {cos_sim:.3f}
- Compatibility probability (Polyvore-style): {prob_val}

Item 1 description (user-provided, may be empty):
{desc1 or "[no description]"}

Item 2 description (user-provided, may be empty):
{desc2 or "[no description]"}

In 3–4 sentences, explain:
1) Whether these pieces likely go well together and why.
2) Any caveats (for example if the model might be biased toward studio product photos).
3) A simple styling tip (shoes, layers, accessories) to make the outfit work better.

Keep the tone concise, friendly, and non-repetitive.
"""
    resp = call_ollama(prompt)
    if resp is None:
        return "Stylist is unavailable (Ollama not running or model not pulled)."
    return resp


def _llm_stylist_for_outfit(
    item_descriptions: List[str],
    outfit_score: float,
) -> str:
    """
    Ask the LLM to comment on a full outfit composed of N generic items.
    item_descriptions: list of strings like ["black crop top", "blue jeans", ...]
    """
    if item_descriptions:
        items_text = "\n".join(
            f"- Item {i+1}: {desc or '[no description]'}"
            for i, desc in enumerate(item_descriptions)
        )
    else:
        items_text = "- [no descriptions provided]"

    prompt = f"""
You are a friendly but honest fashion stylist.

The user has built an outfit composed of these items:

{items_text}

A compatibility model rated the overall outfit with a score of {outfit_score:.2f}
(on a 0–1 scale, where higher means more visually cohesive and compatible).

Please:
1. Briefly describe the overall vibe of this outfit.
2. Point out what works well (colors, silhouettes, style coherence).
3. If the score is below about 0.7, suggest 1–2 concrete tweaks (e.g. "swap the shoes for something lighter", "add a jacket", "tone down the print").
4. Mention one occasion where this outfit would fit nicely (e.g. brunch, office, date night, travel).

Keep it concise (3–5 sentences) and easy to understand.
"""
    resp = call_ollama(prompt)
    if resp is None:
        return "Stylist is unavailable (Ollama not running or model not pulled)."
    return resp


def _llm_stylist_for_item(
    styles: List[dict],
    colors: List[dict],
    desc: str,
) -> str:
    """
    Ask the LLM to summarize style + colors for a single item.
    """
    # Take top 3 styles
    top_styles = styles[:3]
    style_summary = ", ".join(
        f"{s['name']} ({s['prob']*100:.1f}%)" for s in top_styles
    )

    color_summary = ", ".join(c["hex"] for c in colors)

    prompt = f"""
You are a fashion stylist.

I have one clothing item and an ML model has analysed it.

Detected style distribution (approx):
{style_summary}

Dominant colors (hex codes):
{color_summary}

User description of the item (may be empty):
{desc or "[no description]"}

In 3–4 sentences, please:
1) Describe the vibe / style of this item.
2) Suggest what kinds of other pieces it would pair well with (e.g., neutrals, denim, boots, etc.).
3) Mention one scenario where this item would work especially well (e.g., brunch, office, night out).

Keep the tone concise and friendly.
"""
    resp = call_ollama(prompt)
    if resp is None:
        return "Stylist is unavailable (Ollama not running or model not pulled)."
    return resp



# UI and emoticons(were generated using AI to make the app more user friendly)

#tab1, tab2 = st.tabs([" Check Pair", " Outfit Builder & Recommendations"])
tab1, tab2, tab3 = st.tabs(
    [
        "🧩 Build Outfit",
        "🛍️ Find Matches",
        "🎨 Style Analyzer",
    ]
)


# Tab 1: outfit builder & recommendations
with tab1:
    st.subheader("Rate my outfit")
    st.markdown(
        "Upload between **2 and 10 items** (any mix: tops, bottoms, jackets, shoes, bags, etc.) "
        "and we'll compute pairwise compatibilities and an overall outfit score."
    )

    files = st.file_uploader(
        "Upload outfit items",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="outfit_files",
    )

    item_descriptions: List[str] = []

    imgs: List[Image.Image] = []
    if files:
        st.markdown("#### Outfit items")
        cols = st.columns(min(len(files), 4))
        for idx, f in enumerate(files):
            img = Image.open(f).convert("RGB")
            imgs.append(img)

            col = cols[idx % len(cols)]
            with col:
                st.image(img, caption=f"Item {idx+1}", use_container_width=True)
                desc = st.text_input(
                    f"Description for item {idx+1} (optional)",
                    key=f"outfit_desc_{idx}",
                )
                item_descriptions.append(desc)
    else:
        st.info("Upload at least two item images to rate an outfit.")

    # Button created to compute outfit score
    if imgs and len(imgs) >= 2:
        if st.button("Rate this outfit", key="rate_outfit_btn"):
            with st.spinner("Scoring outfit..."):
                probs, outfit_score = score_outfit(
                    clip_model, preprocess, classifier, imgs
                )

                pair_info = None
                if len(imgs) == 2:
                    cos_sim, pair_prob = predict_pair(
                        clip_model, preprocess, classifier, imgs[0], imgs[1]
                    )
                    pair_info = {
                        "cos_sim": float(cos_sim),
                        "prob": float(pair_prob),
                    }

            st.session_state["outfit_results"] = {
                "descs": item_descriptions,
                "probs": probs,
                "score": float(outfit_score),
                "n_items": len(imgs),
                "pair": pair_info, 
            }


    # ---- Stored results ----
    outfit_res = st.session_state.get("outfit_results")
    if outfit_res is not None:
        probs = outfit_res["probs"]
        outfit_score = outfit_res["score"]
        descs = outfit_res["descs"]
        n = outfit_res["n_items"]

        st.markdown("### Outfit results")
        st.metric(
            "Overall outfit compatibility (0–1)",
            f"{outfit_score:.2f}",
        )

        # Interpreting the score in an user-friendly way
        rating_10 = outfit_score * 10.0

        if outfit_score >= 0.75:
            st.success(
                f"High compatibility – this outfit looks cohesive. "
                f"Roughly **{rating_10:.1f}/10** in terms of visual harmony."
            )
        elif outfit_score >= 0.5:
            st.warning(
                f"Moderate compatibility – parts of this outfit work, but there may be some clashes. "
                f"Roughly **{rating_10:.1f}/10**; consider tweaking one piece."
            )
        else:
            st.error(
                f"Low compatibility – the model thinks this combo is quite experimental or mismatched. "
                f"Roughly **{rating_10:.1f}/10**; you might want to swap 1–2 items."
            )

        # A table for pairwise compatibilities
        st.markdown("#### Pairwise compatibility matrix")
        item_labels = [f"Item {i+1}" for i in range(n)]
        import pandas as pd
        df = pd.DataFrame(probs, columns=item_labels, index=item_labels)
        st.dataframe(df.style.format("{:.2f}"))

        # Pair-specific block only if there are exactly 2 items
        pair_info = outfit_res.get("pair")
        if n == 2 and pair_info is not None:
            st.markdown("#### Detailed pair compatibility")

            col_pair = st.columns(2)
            with col_pair[0]:
                st.metric("CLIP cosine similarity", f"{pair_info['cos_sim']:.3f}")
            with col_pair[1]:
                st.metric(
                    "Pair compatibility probability",
                    f"{pair_info['prob']*100:.1f}%",
                )

        st.markdown("##### LLM Stylist")
        if st.checkbox("Ask stylist about this outfit", key="outfit_llm_on"):
            if st.button("Ask stylist for advice", key="outfit_llm_btn"):
                with st.spinner("Stylist is thinking..."):
                    response = _llm_stylist_for_outfit(descs, outfit_score)
                st.markdown("###### Stylist says:")
                st.write(response)

    else:
        st.info("Once you rate an outfit, results will appear here.")


# Tab 2: Finding Matches (shopping helper mode) 
with tab2:
    st.subheader("Find matches for an item")

    st.markdown(
        "Upload an **anchor item** (e.g., a top, shoes, jacket), and we'll suggest "
        "compatible items from the Polyvore test set. Optionally choose a desired style."
    )

    col_anchor = st.columns(2)
    with col_anchor[0]:
        anchor_file = st.file_uploader(
            "Anchor item",
            type=["jpg", "jpeg", "png"],
            key="matches_anchor",
        )
    with col_anchor[1]:
        anchor_desc = st.text_input(
    "Describe this anchor item (optional)", key="matches_anchor_input"
)

    style_options = ["Any", "casual", "formal", "streetwear", "sporty", "minimal", "party"]
    desired_style = st.selectbox(
        "Desired style for matches (optional)", style_options, index=0
    )
    if desired_style == "Any":
        desired_style_arg = None
    else:
        desired_style_arg = desired_style

    # How many candidates to scan from the test pool
    # num_candidates = st.slider(
    #     "How many candidates to scan from the dataset?",
    #     min_value=100,
    #     max_value=600,
    #     value=300,
    #     step=50,
    # )
    num_candidates = 500

    top_k = st.slider(
        "How many top matches to display?",
        min_value=2,
        max_value=10,
        value=4,
        step=1,
    )

    if anchor_file:
        anchor_img = Image.open(anchor_file).convert("RGB")
        st.image(anchor_img, caption="Anchor item", width=260)

        if st.button("Find matches", key="find_matches_btn"):
            with st.spinner("Searching for compatible items..."):
                # To get a candidate pool from the test split
                try:
                    pool = list(get_test_image_pool())
                except Exception:
                    # Fallback
                    pool = sample_test_images(num_candidates * 2)

                if len(pool) > num_candidates:
                    import random

                    candidate_paths = random.sample(pool, num_candidates)
                else:
                    candidate_paths = pool

                ranked = rank_candidates_for_anchor(
                    clip_model,
                    preprocess,
                    classifier,
                    anchor_img,
                    candidate_paths,
                    top_k=top_k,
                    desired_style=desired_style_arg,
                )

            st.session_state["matches_ranked"] = ranked
            st.session_state["matches_anchor_desc"] = anchor_desc or ""

    # Show results if present
    matches_ranked = st.session_state.get("matches_ranked")
    if matches_ranked:
        st.markdown("### Recommended matches")
        n_cols = min(4, len(matches_ranked))
        cols = st.columns(n_cols)

        for idx, item in enumerate(matches_ranked):
            col = cols[idx % n_cols]
            with col:
                img_path = IMAGES_DIR / item["path"]
                if img_path.exists():
                    st.image(img_path, caption=item["path"], use_container_width=True)
                st.write(f"Compatibility prob: **{item['prob']*100:.1f}%**")
                st.write(f"CLIP similarity: `{item['cos_sim']:.3f}`")
                if item["style_score"] is not None:
                    st.write(f"Style alignment: `{item['style_score']:.3f}`")
                st.write(f"Final score: **{item['final_score']:.3f}**")

        # Optional stylist summary
        st.markdown("##### LLM Stylist")
        if st.checkbox("Ask stylist about these matches", key="matches_llm_on"):
            if st.button("Ask stylist", key="matches_llm_btn"):
                avg_prob = sum(m["prob"] for m in matches_ranked) / len(matches_ranked)
                prompt = f"""
You are a fashion stylist.

The user has an anchor item: {st.session_state.get("matches_anchor_desc", "[no description]")}

We suggested {len(matches_ranked)} candidate matches from a catalog, each with a model-based compatibility probability.
The average compatibility probability is about {avg_prob*100:.1f}%.

Please:
1. Briefly describe what kind of pieces these matches tend to be (e.g. casual basics, statement items, etc.).
2. Explain how well they complement the anchor item overall.
3. Suggest what kind of outfit (occasion or vibe) these combinations would work best for.

Keep it concise and friendly.
"""
                with st.spinner("Stylist is thinking..."):
                    resp = call_ollama(prompt)
                # st.markdown("###### Stylist says:")
                st.write(resp)
    else:
        st.info("Upload an anchor item and click 'Find matches' to see recommendations.")


# Tab 3: Style & Color Analyzer
with tab3:
    st.subheader("Analyze style and color for a single item")

    col = st.columns(2)
    with col[0]:
        style_img_file = st.file_uploader(
            "Upload an item image", type=["jpg", "jpeg", "png"], key="style_img"
        )
    with col[1]:
        style_desc = st.text_input(
            "Describe this item (optional)", key="style_desc"
        )

    if style_img_file:
        img = Image.open(style_img_file).convert("RGB")
        st.image(img, caption="Item preview", width=260)

        if st.button("Analyze item", key="analyze_item_btn"):
            with st.spinner("Analyzing style and colors..."):
                styles = analyze_style(clip_model, preprocess, img)
                colors = extract_color_palette(img, n_colors=3)
            st.session_state["style_results"] = {
                "styles": styles,
                "colors": colors,
                "desc": style_desc,
            }

    # Show stored analysis so it survives reruns
    style_res = st.session_state.get("style_results")
    if style_res is not None:
        styles = style_res["styles"]
        colors = style_res["colors"]
        desc = style_res["desc"]

        st.markdown("### Style profile")

        # Show top 3 styles
        top_styles = styles[:3]
        st.write(
            "Top detected styles (approximate, based on CLIP image–text similarity):"
        )
        for s in top_styles:
            st.write(
                f"- **{s['name'].title()}** — {s['prob']*100:.1f}% (raw score {s['score']:.3f})"
            )

        st.markdown("### Color palette")
        color_cols = st.columns(len(colors))
        for c, cc in zip(colors, color_cols):
            with cc:
                r, g, b = c["rgb"]
                hex_code = c["hex"]
                # Colour swatch using markdown + HTML
                swatch_html = f"""
                <div style="
                    width: 60px;
                    height: 60px;
                    border-radius: 8px;
                    border: 1px solid #444;
                    background-color: {hex_code};
                    margin-bottom: 4px;
                "></div>
                """
                st.markdown(swatch_html, unsafe_allow_html=True)
                st.caption(f"{hex_code}\nRGB{c['rgb']}")

        st.markdown("##### LLM Stylist")
        if st.checkbox(
            "Ask stylist how to wear this item", key="style_item_llm_on"
        ):
            if st.button("Ask stylist", key="style_item_llm_btn"):
                with st.spinner("Stylist is thinking..."):
                    resp = _llm_stylist_for_item(styles, colors, desc)
                # st.markdown("###### Stylist says:")
                st.write(resp)
    else:
        st.info("Upload an image and click 'Analyze item' to see style and color information.")

