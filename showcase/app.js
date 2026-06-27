// Titan — Title Review AI — showcase interactions
// ponytail: vanilla JS, no deps. Reveals, nav state, cited-draft explorer with provenance.

const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/* ---- scroll reveals ---- */
const io = new IntersectionObserver(
  (entries) => {
    for (const e of entries) {
      if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); }
    }
  },
  { threshold: 0.15, rootMargin: "0px 0px -8% 0px" }
);
document.querySelectorAll(".reveal").forEach((el) => io.observe(el));

/* ---- nav state ---- */
const nav = document.getElementById("nav");
const onScroll = () => nav.classList.toggle("scrolled", window.scrollY > 24);
addEventListener("scroll", onScroll, { passive: true });
onScroll();

/* ---- bento cursor spotlight ---- */
document.querySelectorAll(".bento .item").forEach((item) => {
  const core = item.querySelector(".core");
  if (!core) return;
  core.style.position = "relative";
  item.addEventListener("pointermove", (e) => {
    const r = item.getBoundingClientRect();
    core.style.setProperty("--mx", `${e.clientX - r.left}px`);
    core.style.setProperty("--my", `${e.clientY - r.top}px`);
  });
});

/* ---- cited-draft explorer ----
   Sample output for a real-shaped commitment. Citations carry data-ref that maps
   to a retrieved-evidence snippet, demonstrating end-to-end provenance. */
const C = (ref, label) => `<span class="cite" data-ref="${ref}">[${label}]</span>`;

const SECTIONS = [
  {
    name: "Vesting", tag: "ALTA · Schedule A",
    body: `Fee simple title to the land is currently vested in <b>Margaret A. Whitfield</b>, a single woman, by virtue of a Warranty Deed recorded as Instrument No. 2019-0044187 ${C("v1", "commitment · p3")}. No subsequent conveyance out of the vested owner appears of record ${C("v2", "deed_0044187 · p1")}.`,
    ev: [
      { ref: "v1", doc: "wayne_county_commitment", page: 3, snip: `Title to the estate or interest in the land is at the Commitment Date vested in: <mark>Margaret A. Whitfield, a single woman.</mark>` },
      { ref: "v2", doc: "deed_0044187", page: 1, snip: `…does hereby grant and warrant unto <mark>Margaret A. Whitfield</mark> the following described premises…` },
    ],
  },
  {
    name: "Legal Description", tag: "ALTA · Schedule A",
    body: `The land is described as <b>Lot 14, Block 7 of the Riverside Heights Subdivision</b>, according to the recorded plat thereof ${C("l1", "commitment · p4")}, situated in Wayne County. The description matches the plat dimensions of record with no apparent gap or overlap ${C("l2", "plat_RH_1962 · p1")}.`,
    ev: [
      { ref: "l1", doc: "wayne_county_commitment", page: 4, snip: `<mark>Lot 14, Block 7, Riverside Heights Subdivision</mark>, as recorded in Liber 88 of Plats, Page 21, Wayne County Records.` },
      { ref: "l2", doc: "plat_RH_1962", page: 1, snip: `Lot 14 — 60.00 ft frontage on Riverside Drive, depth <mark>120.00 ft</mark>, Block 7.` },
    ],
  },
  {
    name: "Chain of Title", tag: "ALTA · 24-month search",
    body: `The chain is unbroken for the search period: Hartman → Doyle (2014) ${C("c1", "deed_2014-991 · p1")}, Doyle → Whitfield (2019) ${C("c2", "deed_0044187 · p1")}. Each conveyance was recorded within the statutory window and references the prior instrument ${C("c3", "commitment · p5")}.`,
    ev: [
      { ref: "c1", doc: "deed_2014-991", page: 1, snip: `Grantor: <mark>Robert J. Hartman</mark>; Grantee: Susan E. Doyle. Recorded 03/18/2014.` },
      { ref: "c2", doc: "deed_0044187", page: 1, snip: `Grantor: <mark>Susan E. Doyle</mark>; Grantee: Margaret A. Whitfield. Recorded 06/02/2019.` },
      { ref: "c3", doc: "wayne_county_commitment", page: 5, snip: `Being the same premises conveyed by instrument <mark>2014-000991</mark>.` },
    ],
  },
  {
    name: "Open Encumbrances", tag: "ALTA · Schedule B-II",
    body: `One open mortgage remains of record: a Mortgage to <b>Great Lakes Savings Bank</b> securing $182,000, recorded as Instrument No. 2019-0044188 ${C("e1", "mortgage_44188 · p1")}. No release or satisfaction has been recorded ${C("e2", "commitment · p6")}.`,
    ev: [
      { ref: "e1", doc: "mortgage_44188", page: 1, snip: `Mortgagor: Margaret A. Whitfield. Mortgagee: <mark>Great Lakes Savings Bank</mark>. Principal: $182,000.00.` },
      { ref: "e2", doc: "wayne_county_commitment", page: 6, snip: `Mortgage recorded as 2019-0044188 <mark>remains open of record</mark>; no discharge found.` },
    ],
  },
  {
    name: "Easements", tag: "ALTA · Schedule B-II",
    body: `A perpetual utility easement runs along the rear 10 feet of the parcel in favor of Wayne County Electric ${C("ea1", "easement_8841 · p2")}. The easement is shown on the recorded plat and does not encroach on the principal structure ${C("ea2", "plat_RH_1962 · p1")}.`,
    ev: [
      { ref: "ea1", doc: "easement_8841", page: 2, snip: `…a <mark>10-foot utility easement</mark> across the rear lot line in favor of Wayne County Electric Cooperative.` },
      { ref: "ea2", doc: "plat_RH_1962", page: 1, snip: `Rear setback shown as <mark>10' P.U.E.</mark> (public utility easement).` },
    ],
  },
  {
    name: "Schedule B-I", tag: "Requirements",
    body: `To insure, the company requires: (1) a recorded discharge of the Great Lakes Savings Bank mortgage if to be paid at closing ${C("r1", "commitment · p7")}, and (2) a Statement of Authority or current grantor signature confirming marital status ${C("r2", "commitment · p7")}.`,
    ev: [
      { ref: "r1", doc: "wayne_county_commitment", page: 7, snip: `Requirement 3: <mark>Record satisfaction of mortgage</mark> instrument 2019-0044188.` },
      { ref: "r2", doc: "wayne_county_commitment", page: 7, snip: `Requirement 5: Provide evidence of <mark>marital status</mark> of vested owner at date of conveyance.` },
    ],
  },
  {
    name: "Schedule B-II", tag: "Exceptions",
    body: `Standard exceptions apply for rights of parties in possession and matters a survey would disclose ${C("x1", "commitment · p8")}. Special exception is taken for the recorded easement and the open mortgage described above ${C("x2", "commitment · p8")}.`,
    ev: [
      { ref: "x1", doc: "wayne_county_commitment", page: 8, snip: `Exception 1: <mark>Rights of parties in possession</mark> not shown by the public records.` },
      { ref: "x2", doc: "wayne_county_commitment", page: 8, snip: `Exception 6: Easement per 8841 and Mortgage per <mark>2019-0044188</mark>.` },
    ],
  },
  {
    name: "Taxes & Survey", tag: "ALTA/NSPS",
    body: `Real property taxes for the current year are paid through the second installment; no delinquency appears ${C("t1", "tax_cert_2024 · p1")}. The ALTA/NSPS survey shows the dwelling within all setbacks with no observed encroachments ${C("t2", "survey_2024 · p1")}.`,
    ev: [
      { ref: "t1", doc: "tax_cert_2024", page: 1, snip: `Parcel 82-014-007: 2024 taxes <mark>PAID — no delinquent balance.</mark>` },
      { ref: "t2", doc: "survey_2024", page: 1, snip: `Improvements located <mark>within recorded setbacks</mark>; no encroachments observed.` },
    ],
  },
];

(function explorer() {
  const tabs = document.getElementById("sectionTabs");
  const nameEl = document.getElementById("secName");
  const tagEl = document.getElementById("secTag");
  const bodyEl = document.getElementById("secBody");
  const evEl = document.getElementById("evidence");
  if (!tabs) return;

  // build tabs
  SECTIONS.forEach((s, i) => {
    const b = document.createElement("button");
    b.className = "sec-chip";
    b.textContent = s.name;
    b.setAttribute("role", "tab");
    b.setAttribute("aria-selected", i === 0 ? "true" : "false");
    b.addEventListener("click", () => select(i));
    tabs.appendChild(b);
  });

  function renderEvidence(s, activeRef) {
    evEl.innerHTML = s.ev
      .map(
        (e) => `<div class="ev-doc${e.ref === activeRef ? " lit" : ""}" data-ref="${e.ref}">
          <div class="ref"><span>${e.doc}</span><span class="pg">page ${e.page}</span></div>
          <div class="snip">${e.snip}</div>
        </div>`
      )
      .join("");
  }

  function wireCites(s) {
    bodyEl.querySelectorAll(".cite").forEach((c) => {
      const ref = c.dataset.ref;
      const lit = () => {
        bodyEl.querySelectorAll(".cite").forEach((x) => x.classList.toggle("active", x.dataset.ref === ref));
        evEl.querySelectorAll(".ev-doc").forEach((d) => d.classList.toggle("lit", d.dataset.ref === ref));
        const target = evEl.querySelector(`.ev-doc[data-ref="${ref}"]`);
        if (target) target.scrollIntoView({ block: "nearest", behavior: reduce ? "auto" : "smooth" });
      };
      c.addEventListener("mouseenter", lit);
      c.addEventListener("focus", lit);
      c.addEventListener("click", lit);
      c.tabIndex = 0;
    });
  }

  let current = -1;
  function select(i) {
    if (i === current) return;
    current = i;
    const s = SECTIONS[i];
    [...tabs.children].forEach((c, j) => c.setAttribute("aria-selected", j === i ? "true" : "false"));

    bodyEl.classList.add("fade");
    setTimeout(() => {
      nameEl.textContent = s.name;
      tagEl.textContent = s.tag;
      bodyEl.innerHTML = s.body;
      renderEvidence(s, null);
      wireCites(s);
      bodyEl.classList.remove("fade");
    }, reduce ? 0 : 180);
  }

  select(0);
})();
