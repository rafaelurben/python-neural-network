import tkinter

class TrainingGUI(tkinter.Tk):
    def __init__(self, trainer, alwaysontop=True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._nn_trainer = trainer

        self.title("Neural Training GUI - by rafaelurben")
        self.resizable(False, False)
        self.configure(background='#000')

        self._nn_interrupted = False

        # Settings

        label = tkinter.Label(self, text="Settings", bg='#000', fg='#FFF')
        label.configure(font=("Arial", 20))
        label.grid(row=0, column=0, columnspan=4, sticky="n")

        self._nn_entryfields = {}

        for i, name in enumerate(trainer.EDITABLE_FIELDS):

            label = tkinter.Label(self, text=name, bg='#000', fg='#FFF')
            label.configure(font=("Arial", 12))
            label.grid(row=i+1, column=0, columnspan=2, sticky="w", padx=10)

            entry = tkinter.Entry(self)
            entry.insert(0, str(getattr(trainer, name)))
            entry.grid(row=i+1, column=2, columnspan=2, sticky="e", padx=10)

            self._nn_entryfields[name] = entry

        h = len(trainer.EDITABLE_FIELDS)+2

        btn = tkinter.Button(self, text="Save", command=self._nn_save)
        btn.grid(row=h, column=2, columnspan=2, sticky="ne", pady=10, padx=10)
        h += 1

        # Runner

        label = tkinter.Label(self, text="Runner", bg='#000', fg='#FFF')
        label.configure(font=("Arial", 20))
        label.grid(row=h, column=0, columnspan=4, sticky="n")
        h += 1

        label = tkinter.Label(self, text="Runs", bg='#000', fg='#FFF')
        label.configure(font=("Arial", 12))
        label.grid(row=h, column=0, columnspan=2, sticky="w", padx=10)

        self._nn_e_runs = tkinter.Entry(self)
        self._nn_e_runs.insert(0, "0")
        self._nn_e_runs.grid(row=h, column=2, columnspan=2, sticky="e", padx=10)
        h += 1

        self._nn_b_run_once = tkinter.Button(self, text="Run once", command=self._nn_run_once)
        self._nn_b_run_once.grid(row=h, column=0, sticky="w", pady=10, padx=10)
        self._nn_b_run_infinite = tkinter.Button(self, text="Run infinite", command=self._nn_run_infinite)
        self._nn_b_run_infinite.grid(row=h, column=1, sticky="w", pady=10, padx=0)
        self._nn_b_run_entry_times = tkinter.Button(self, text="Run n times", command=self._nn_run_entry_times)
        self._nn_b_run_entry_times.grid(row=h, column=2, sticky="w", pady=10, padx=0)
        self._nn_b_interrupt = tkinter.Button(self, text="Interrupt", command=self._nn_interrupt, state="disabled")
        self._nn_b_interrupt.grid(row=h, column=3, sticky="e", pady=10, padx=10)
        h += 1

        # Data

        label = tkinter.Label(self, text="Data", bg='#000', fg='#FFF')
        label.configure(font=("Arial", 20))
        label.grid(row=h, column=0, columnspan=4, sticky="n")
        h += 1

        label = tkinter.Label(self, text="Generation", bg='#000', fg='#FFF')
        label.configure(font=("Arial", 12))
        label.grid(row=h, column=0, columnspan=2, sticky="w", padx=10)

        self._nn_l_generation = tkinter.Label(self, text=str(trainer.generation), bg='#000', fg='#FFF')
        self._nn_l_generation.configure(font=("Arial", 12))
        self._nn_l_generation.grid(row=h, column=2, columnspan=2, sticky="w", padx=10)
        h += 1

        label = tkinter.Label(self, text="Latest Score", bg='#000', fg='#FFF')
        label.configure(font=("Arial", 12))
        label.grid(row=h, column=0, columnspan=2, sticky="w", padx=10)

        self._nn_l_score = tkinter.Label(self, text="Not run yet", bg='#000', fg='#FFF')
        self._nn_l_score.configure(font=("Arial", 12))
        self._nn_l_score.grid(row=h, column=2, columnspan=2, sticky="w", padx=10)
        h += 1

        label = tkinter.Label(self, text="State", bg='#000', fg='#FFF')
        label.configure(font=("Arial", 12))
        label.grid(row=h, column=0, columnspan=2, sticky="w", padx=10)

        self._nn_l_state = tkinter.Label(self, text="Idle.", bg='#000', fg='#FFF')
        self._nn_l_state.configure(font=("Arial", 12))
        self._nn_l_state.grid(row=h, column=2, columnspan=2, sticky="w", padx=10)
        h += 1
        
        label = tkinter.Label(self, text="Folder", bg='#000', fg='#FFF')
        label.configure(font=("Arial", 12))
        label.grid(row=h, column=0, columnspan=2, sticky="w", padx=10)

        self._nn_l_folder = tkinter.Label(self, text=trainer.folder, bg='#000', fg='#FFF')
        self._nn_l_folder.configure(font=("Arial", 12))
        self._nn_l_folder.grid(row=h, column=2, columnspan=2, sticky="w", padx=10)
        h += 1
        
        label = tkinter.Label(self, text="Name", bg='#000', fg='#FFF')
        label.configure(font=("Arial", 12))
        label.grid(row=h, column=0, columnspan=2, sticky="w", padx=10)

        self._nn_l_name = tkinter.Label(self, text=trainer.name, bg='#000', fg='#FFF')
        self._nn_l_name.configure(font=("Arial", 12))
        self._nn_l_name.grid(row=h, column=2, columnspan=2, sticky="w", padx=10)
        h += 1

        # Always on top
        if alwaysontop:
            self.call('wm', 'attributes', '.', '-topmost', '1')

    def _nn_interrupt(self):
        self._nn_interrupted = True
        print("Interrupting...")

    def _nn_save(self):
        for name, entry in self._nn_entryfields.items():
            tp = type(getattr(self._nn_trainer, name))
            setattr(self._nn_trainer, name, tp(entry.get()))
        print("Updated settings!")

    def nn_afterrun(self):
        ...

    def _nn_run_n_times(self, count):
        self._nn_interrupted = False

        self._nn_l_state.configure(text="Training next...")

        self._nn_e_runs.config(state="disabled")
        self._nn_b_interrupt.config(state="normal")
        self._nn_b_run_once.config(state="disabled")
        self._nn_b_run_entry_times.config(state="disabled")
        self._nn_b_run_infinite.config(state="disabled")
        self.update()

        while count > 0:
            if self._nn_interrupted:
                print("Interrupted!")
                break

            self._nn_e_runs.config(state="normal")
            self._nn_e_runs.delete(0, tkinter.END)
            self._nn_e_runs.insert(0, str(count))
            self._nn_e_runs.config(state="disabled")
            self.update()

            highscore = self._nn_trainer.run_generation()
            generation = self._nn_trainer.generation

            self._nn_l_generation.configure(text=str(generation))
            self._nn_l_score.configure(text=str(highscore))
            self.update()

            self.nn_afterrun()

            count -= 1


        self._nn_e_runs.config(state="normal")
        self._nn_b_interrupt.config(state="disabled")
        self._nn_b_run_once.config(state="normal")
        self._nn_b_run_entry_times.config(state="normal")
        self._nn_b_run_infinite.config(state="normal")

        self._nn_l_state.configure(text="Idle.")
        self._nn_e_runs.delete(0, tkinter.END)
        self._nn_e_runs.insert(0, "0")
        self.update()

    def _nn_run_once(self):
        self._nn_run_n_times(1)

    def _nn_run_entry_times(self):
        count = int(self._nn_e_runs.get())
        self._nn_run_n_times(count)

    def _nn_run_infinite(self):
        self._nn_run_n_times(float("inf"))
