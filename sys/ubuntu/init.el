;;; package --- macOS init.el

;; Author:  Jeffrey Liu
;; Version: 2023.11.22.16

;;; Commentary:

;;; Code:

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(eval-when-compile
  (require 'use-package))

(use-package package
  :ensure nil
  :config
  (setq package-enable-at-startup nil)
  (add-to-list 'package-archives
    '("melpa" . "https://melpa.org/packages/")
    'append))

(use-package files
  :ensure nil
  :config
  (setq confirm-kill-processes nil)
  (setq create-lockfiles nil)
  (setq make-backup-files nil))

(use-package dired
  :ensure nil
  :commands
  (dired-get-file-for-visit
   dired-find-alternate-file)
  :after
  (ace-window)
  :init
  (defun dired-ace-window-jump ()
    (interactive)
    (let ((value (dired-get-file-for-visit))
          (current-window (selected-window)))
      (cond
       ((> (length (aw-window-list)) 2)
        (aw-switch-to-window (aw-select nil))
        (funcall 'find-file value))
       (t (call-interactively 'dired-find-file-other-window)))))
  (defun dired-ace-window-show ()
    (interactive)
    (let ((value (dired-get-file-for-visit))
          (current-window (selected-window)))
      (cond
       ((> (length (aw-window-list)) 2)
        (aw-switch-to-window (aw-select nil))
        (funcall 'find-file value))
       (t (call-interactively 'dired-find-file-other-window)))
      (unless (eq current-window (selected-window))
        (aw-flip-window))))
  (defun dired-select-show ()
    (interactive)
    (let ((value (dired-get-file-for-visit)))
      (if (file-directory-p value)
          (dired-find-alternate-file)
        (dired-ace-window-jump))))
  :bind
  ("C-x C-d" . dired)
  (:map dired-mode-map
        ("i" . (lambda() (interactive) (find-alternate-file "..")))
        ("o" . dired-ace-window-jump)
        ("C-o" . dired-ace-window-show)
        ("RET" . dired-select-show))
  :hook
  (dired-after-readin . (lambda()
                          (rename-buffer
                            (generate-new-buffer-name dired-directory))))
  :config
  (setq dired-listing-switches "-alho --group-directories-first")
  (put 'dired-find-alternate-file 'disabled nil))

(use-package sh-script
  :ensure nil
  :delight
  (sh-mode))

(use-package auto-package-update
  :ensure t
  :config
  (setq auto-package-update-delete-old-versions t)
  (setq auto-package-update-hide-results t)
  (auto-package-update-maybe))

(custom-set-variables
 ;; custom-set-variables was added by Custom.
 ;; If you edit it by hand, you could mess it up, so be careful.
 ;; Your init file should contain only one such instance.
 ;; If there is more than one, they won't work right.
 '(package-selected-packages
   '(pyvenv python-black flycheck yasnippet-snippets yasnippet ace-window
            markdown-mode markdown delight which-key helm auto-package-update)))
(custom-set-faces
 ;; custom-set-faces was added by Custom.
 ;; If you edit it by hand, you could mess it up, so be careful.
 ;; Your init file should contain only one such instance.
 ;; If there is more than one, they won't work right.
 )

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; :commands  defer  - run this package, use defer instead
;; :hook      defer  - other mode hook to this
;; :bind      defer  - binding keys
;; :defer t          - replace commands for defer
;; :init      before - always run, must have a one above
;; :config    after
;; :delight   hide or abbr mode-line name
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; C-c a    : ace-window jump
;; C-c b    : python black formater
;; C-c e    : buffer end
;; C-c h    : buffer begin
;; C-c m    : Set Mark
;; C-c f    : flycheck error list
;; C-c s    : ace-window swap
;; C-c y    : YASnippets
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(use-package delight
  :ensure t)

(use-package whitespace
  :ensure t
  :delight
  (global-whitespace-mode)
  :config
  (setq whitespace-style '(face empty tabs lines-tail trailing))
  (global-whitespace-mode t))

(use-package helm
  :ensure t
  :bind
  (("M-x" . helm-M-x)))

(use-package which-key
  :ensure t
  :delight
  (which-key-mode)
  :config
  (which-key-mode))

(use-package ace-window
  :ensure t
  :commands
  (aw-window-list
   aw-switch-to-window
   aw-select
   aw-flip-window)
  :demand t
  :bind
  ("C-c a" . ace-window)
  ("C-c s" . ace-swap-window)
  :config
  (setq aw-keys '(?a ?s ?d ?f ?g ?h ?j ?k ?w ?o))
  (setq aw-scope 'frame)
  (setq aw-dispatch-always t))

(use-package ibuffer
  :ensure t
  :commands
  (ibuffer-current-buffer
   ibuffer-switch-to-saved-filter-groups
   ibuffer-auto-mode)
  :init
  (defun ibuffer-ace-window-jump ()
    (interactive)
    (let ((buf (ibuffer-current-buffer t)))
      (bury-buffer (current-buffer))
      (cond
       ((> (length (aw-window-list)) 2)
        (aw-switch-to-window (aw-select nil))))
      (switch-to-buffer buf)))
  (defun ibuffer-ace-window-show ()
    (interactive)
    (let ((buf (ibuffer-current-buffer t))
          (current-window (selected-window)))
      (bury-buffer (current-buffer))
      (cond
       ((> (length (aw-window-list)) 2)
        (aw-switch-to-window (aw-select nil))))
      (switch-to-buffer buf)
      (unless (eq current-window (selected-window))
        (aw-flip-window))))
  (defvar ibuffer-show-empty-filter-groups nil)
  (defvar ibuffer-saved-filter-groups
    (quote (("Default"
             ("dir" (mode . dired-mode))
             ("py" (mode . python-mode))
             ("md" (mode . markdown-mode))
             ("sh" (mode . sh-mode))
             ("rst" (mode . rst-mode))
             ("pip" (name . "pip.*.txt"))
             ("cfg" (or
                     (mode . org-mode)
                     (name . "init.el")
                     (name . ".zshrc")
                     (name . ".gitignore")))
             ("esh" (mode . eshell-mode))
             ("eww" (mode . eww-mode))
             ("lsp" (or
                     (name . "\\*lsp.*")
                     (name . "\\*pylsp.*")
                     (name . "\\*ts-ls.*")
                     (name . "\\*eslint.*")))))))
  :bind
  ("C-x C-b" . ibuffer)
  ("C-c w p" . previous-buffer)
  ("C-c w n" . next-buffer)
  (:map ibuffer-mode-map
        ("o" . ibuffer-ace-window-jump)
        ("C-o" . ibuffer-ace-window-show)
        ("RET" . ibuffer-ace-window-jump))
  :hook
  (ibuffer-mode . (lambda()
                    (ibuffer-switch-to-saved-filter-groups "Default")
                    (ibuffer-auto-mode 1)))
  :config
  (setq ibuffer-display-summary nil))

(use-package company
  :ensure t
  :delight
  (company-mode)
  :hook
  (python-mode . company-mode)
  (markdown-mode . company-mode)
  (sh-mode . company-mode)
  (emacs-lisp-mode . company-mode)
  :config
  (setq company-idle-delay 0)
  (setq company-minimum-prefix-length 1)
  (setq company-tooltip-limit 20))

(use-package flycheck
  :ensure t
  :delight
  (flycheck-mode)
  :bind
  ("C-c f" . flycheck-list-errors)
  :hook
  (python-mode . flycheck-mode)
  (emacs-lisp-mode . flycheck-mode))

(use-package markdown-mode
  :ensure t
  :defer t
  :delight
  (markdown-mode "Md"))

(use-package python-black
  :ensure t
  :bind
  ("C-c b" . python-black-buffer)
  :config
  (setq python-black-extra-args '("-l" "79" "-t" "py37")))

(use-package yasnippet
  :ensure t
  :commands
  (yas-reload-all)
  :delight
  (yas-minor-mode)
  :bind
  (:map yas-minor-mode-map
        ("C-c y" . yas-expand))
  :hook
  (python-mode . yas-minor-mode)
  (markdown-mode . yas-minor-mode)
  :config
  (yas-reload-all))

(use-package yasnippet-snippets
  :ensure t
  :defer t)

(use-package pyvenv
  :ensure t
  :config
  (pyvenv-mode t)
  (pyvenv-activate "~/pyvenv/dev311/venv"))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(use-package emacs
  :commands
  (uhd)
  :init
  (defun uhd()
    "init window layout uhd"
    (interactive)
    (split-window-right 82)
    (windmove-right)
    (split-window-below -15)
    (split-window-right 82)
    (windmove-right)
    (split-window-below)
    (windmove-down)
    (windmove-down)
    (ibuffer)
    (windmove-left)
    )
  :bind
  ("C-c h" . beginning-of-buffer)
  ("C-c e" . end-of-buffer)
  ("C-c m" . set-mark-command)
  ("C-c w a" . winner-undo)
  ("C-c w e" . winner-redo)
  :hook
  (find-file . read-only-mode)
  :config
  (menu-bar-mode -1)
  (tool-bar-mode -1)
  (electric-pair-mode 1)
  (show-paren-mode 1)
  (winner-mode 1)
  (line-number-mode 1)
  (column-number-mode 1)
  (global-visual-line-mode 1)
  (global-auto-revert-mode 1)
  (indent-tabs-mode -1)
  (global-hl-line-mode 1)
  (setq tab-width 2)
  (setq inhibit-startup-message t)
  (setq initial-scratch-message ";;; Elisp")
  (setq frame-background-mode 'dark)
  (setq mode-line-frame-identification "")
  (setq display-buffer-alist
        '(("^~/.*/$\\|^/.*/$\\|\\*Flycheck errors\\*"
           (display-buffer-reuse-window
               display-buffer-in-side-window)
           (side            . bottom)
           (reusable-frames . visible)
           (window-height   . 0.20))
          ("\\*compilation\\*"
           display-buffer-no-window
           (allow-no-window . t))))
  (uhd)
  :delight
  (visual-line-mode)
  (emacs-lisp-mode "EL")
  (rst-mode "RST")
  (python-mode "Py")
  (lisp-interaction-mode "LI")
  (eldoc-mode))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(provide 'init)
;;; init.el ends here
