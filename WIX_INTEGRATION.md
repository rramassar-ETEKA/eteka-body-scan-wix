# Integration dans Wix

## URL d'integration

Utilisez **toujours** l'URL avec `?embed=1` pour l'integration Wix :

```
https://eteka-body-scan-wix.vercel.app?embed=1
```

Le parametre `?embed=1` :
- Cache le header et footer (votre site Wix garde ses propres elements)
- Met le fond transparent (l'app prend les couleurs de votre page Wix)
- Active le **redimensionnement automatique** de l'iframe (postMessage)

## Methode 1 : Embed HTML (recommandee)

1. Dans l'editeur Wix, cliquez sur **+ Ajouter** > **Embed Code** > **Embed HTML** (ou **Code HTML personnalise**)
2. Selectionnez **HTTP Embed** ou **Code**
3. Collez le code suivant :

```html
<iframe
  id="eteka-bodyscan-iframe"
  src="https://eteka-body-scan-wix.vercel.app?embed=1"
  width="100%"
  height="900"
  frameborder="0"
  allow="camera; microphone; fullscreen"
  style="border: none; display: block; background: transparent;"
></iframe>

<script>
  // Auto-resize iframe based on content height
  window.addEventListener("message", function (event) {
    if (event.data && event.data.type === "eteka-bodyscan-resize") {
      var iframe = document.getElementById("eteka-bodyscan-iframe");
      if (iframe && event.data.height) {
        iframe.style.height = (event.data.height + 20) + "px";
      }
    }
  });
</script>
```

## Methode 2 : Velo (Dev Mode)

Si vous etes en mode developpeur Wix :

1. Ajoutez un element **HTML iframe** sur votre page (ID par defaut `#html1`)
2. Dans le panneau Code en bas :

```javascript
$w.onReady(function () {
  $w("#html1").src = "https://eteka-body-scan-wix.vercel.app?embed=1";

  // Auto-resize iframe
  $w("#html1").onMessage((event) => {
    if (event.data && event.data.type === "eteka-bodyscan-resize") {
      $w("#html1").height = event.data.height + 20;
    }
  });
});
```

## Methode 3 : Bouton vers nouvelle fenetre (free plan)

Si Wix bloque l'embed (plan gratuit), proposez l'app dans un nouvel onglet :

1. Ajoutez un **bouton** "Scanner ma morphologie"
2. Lien : `https://eteka-body-scan-wix.vercel.app`
3. Ouvrir dans : **Nouvelle fenetre**

## Recommandations design

Pour une integration visuelle propre :
- Mettez l'iframe dans une **section sombre** de votre page Wix (le theme de l'app est sombre)
- Hauteur initiale recommandee : **900px** (l'auto-resize ajuste apres chargement)
- Largeur : **100%** (l'app est responsive)

## Verification

Apres integration, testez :
- L'app se charge sans header ETEKA en double
- L'iframe se redimensionne automatiquement quand le contenu change
- Le fond est transparent (les couleurs Wix se voient autour)
- La camera demande les permissions correctement

## Limitation `allow="camera"`

Sur certains navigateurs (notamment Safari mobile), l'attribut `allow="camera"` dans
l'iframe Wix peut etre limite. Dans ce cas, l'utilisateur peut toujours utiliser le
mode **Upload** au lieu de la camera directe.

## URL alternative pour test

Pour tester l'app en plein ecran (avec header/footer) :
```
https://eteka-body-scan-wix.vercel.app
```
