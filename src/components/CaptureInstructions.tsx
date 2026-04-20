"use client";

interface CaptureInstructionsProps {
  onContinue: () => void;
  onBack: () => void;
}

const RULES = [
  {
    title: "Fond neutre",
    desc: "Placez-vous devant un mur uni (blanc, gris clair) sans motifs ni objets.",
  },
  {
    title: "Vetements moulants",
    desc: "Portez un short / legging et une brassiere / top ajuste. Evitez les vetements amples.",
  },
  {
    title: "Bras ecartes du corps",
    desc: "Laissez un espace entre les bras et le torse (angle de 30 a 45 degres). Indispensable pour separer les zones a mesurer.",
  },
  {
    title: "Pieds ecartes a la largeur des epaules",
    desc: "Stable, droit, sans pencher.",
  },
  {
    title: "Bonne lumiere uniforme",
    desc: "Evitez le contre-jour et les ombres fortes sur le corps.",
  },
  {
    title: "Corps entier dans le cadre",
    desc: "De la tete aux pieds, prise a environ 2 metres de distance.",
  },
  {
    title: "Telephone a hauteur de la taille",
    desc: "Posez le telephone sur un support a ~1m du sol pour une photo droite.",
  },
];

export default function CaptureInstructions({ onContinue, onBack }: CaptureInstructionsProps) {
  return (
    <div className="max-w-2xl mx-auto space-y-5">
      <div className="text-center space-y-1">
        <h2 className="text-xl font-bold">Conseils pour des mesures precises</h2>
        <p className="text-sm text-[var(--foreground)]/60">
          Suivez ces consignes pour obtenir les meilleurs resultats
        </p>
      </div>

      <div className="glass rounded-2xl p-5 space-y-4">
        {RULES.map((rule, idx) => (
          <div key={idx} className="flex items-start gap-3">
            <div className="flex-shrink-0 w-7 h-7 rounded-full bg-[var(--primary)]/20 text-[var(--primary-light)] flex items-center justify-center text-sm font-semibold">
              {idx + 1}
            </div>
            <div className="flex-1">
              <div className="font-semibold text-[var(--foreground)]">{rule.title}</div>
              <div className="text-sm text-[var(--foreground)]/60 mt-0.5">{rule.desc}</div>
            </div>
          </div>
        ))}
      </div>

      <div className="glass rounded-xl p-4 border-l-4 border-[var(--warning)]">
        <div className="text-sm font-semibold text-[var(--warning)] mb-1">
          Important
        </div>
        <div className="text-sm text-[var(--foreground)]/80">
          Si les bras touchent le corps, il sera impossible de mesurer separement
          la poitrine, la taille, les hanches et les biceps avec precision.
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3 pt-2">
        <button
          onClick={onBack}
          className="py-3 rounded-xl border border-[var(--border)] text-[var(--foreground)]/70 hover:bg-[var(--surface-light)] transition-all"
        >
          Retour
        </button>
        <button
          onClick={onContinue}
          className="py-3 rounded-xl font-semibold text-white bg-[var(--primary)] hover:bg-[var(--primary-light)] active:scale-[0.98] transition-all"
        >
          J&apos;ai compris, commencer
        </button>
      </div>
    </div>
  );
}
