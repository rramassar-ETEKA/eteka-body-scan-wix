# Integration dans Wix

## Methode 1 : Element HTML (recommandee, la plus simple)

1. Dans votre editeur Wix, cliquez sur **Ajouter** > **Embed Code** > **Embed HTML** (ou **Incorporer HTML**)
2. Choisissez **Embed a site** (Incorporer un site)
3. Collez cette URL :

```
https://eteka-body-scan-wix.vercel.app
```

4. Ajustez la largeur et hauteur (recommande : pleine largeur, hauteur 900-1200px)

## Methode 2 : Code iframe custom

Pour plus de controle (hauteur dynamique, styling), ajoutez un element **HTML Embed**
> **HTML Code** et collez :

```html
<iframe
  src="https://eteka-body-scan-wix.vercel.app"
  width="100%"
  height="1100"
  frameborder="0"
  allow="camera; microphone; fullscreen"
  style="border: none; display: block;"
></iframe>
```

**Important** : l'attribut `allow="camera"` est necessaire pour que la camera fonctionne
depuis l'iframe.

## Methode 3 : Velo (code Wix developpeur)

Si vous avez Wix avec Velo (Dev Mode) :

1. Activez **Developer Tools** > **Enable Dev Mode**
2. Ajoutez un element `#html1` (Custom Element ou HTML iframe)
3. Dans le code de page :

```javascript
$w.onReady(function () {
  $w("#html1").src = "https://eteka-body-scan-wix.vercel.app";
});
```

## Verification

Apres integration :
- Ouvrez votre page Wix en mode apercu
- L'app doit se charger dans l'iframe
- La camera doit demander les permissions
- Le bouton "Analyser" doit fonctionner

## Depannage

**La camera ne fonctionne pas :**
- Verifiez que votre site Wix est en HTTPS
- Ajoutez `allow="camera; microphone"` dans le code iframe

**L'iframe ne s'affiche pas :**
- Les free plans Wix limitent l'embed code. Passez a un plan payant Wix ou
  contactez-nous pour un deploiement alternatif.

**Page blanche dans l'iframe :**
- Verifiez que l'URL est bien accessible dans un navigateur
- Inspectez la console pour les erreurs CSP
