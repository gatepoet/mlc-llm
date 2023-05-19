//
//  DownloadView.swift
//  MLCChat
//
//  Created by Yaxing Cai on 5/11/23.
//

import SwiftUI

struct StartView: View {
    @EnvironmentObject var state: StartState
    @State private var isAdding: Bool = false
    @State private var inputModelUrl: String = ""
    
    var body: some View {
        NavigationStack {
            List{
                Section(header: Text("Models")){
                    ForEach(state.models) { modelState in
                        ModelView().environmentObject(modelState)
                    }
                    if !isAdding {
                        Button("Add model") {
                            isAdding = true
                        }.buttonStyle(.borderless)
                    } else {
                        TextField("Model URL", text: $inputModelUrl)
                    }
                }
            }.navigationTitle("MLC Chat")
            .toolbar{
                if isAdding {
                    ToolbarItem(placement: .navigationBarLeading) {
                        Button("Cancel") {
                            isAdding = false
                            inputModelUrl = ""
                        }
                        .opacity(0.9)
                        .padding()
                    }
                    if !inputModelUrl.isEmpty {
                        ToolbarItem(placement: .navigationBarTrailing) {
                            Button("Add") {
                                state.addModel(modelRemoteBaseUrl: inputModelUrl)
                                isAdding = false
                                inputModelUrl = ""
                            }
                            .opacity(0.9)
                            .padding()
                        }
                    }
                }
            }
        }
    }
}


struct StartView_Previews: PreviewProvider {
    static var previews: some View {
        StartView().environmentObject(StartState())
    }
}
